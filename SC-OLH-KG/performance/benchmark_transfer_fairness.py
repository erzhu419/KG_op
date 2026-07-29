#!/usr/bin/env python3
"""Run one transfer-CBO row under an immutable source/target contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive  # noqa: E402
from baselines.transfer_bo_adapters import (  # noqa: E402
    TRANSFER_METHODS,
    TransferBOConfig,
    TransferConstrainedBO,
)
from core.designs import (  # noqa: E402
    load_frozen_source_informed_design,
)
from core.terminal_verification import (  # noqa: E402
    freeze_objective_incumbent_shortlist,
    parse_verification_candidate_budgets,
    select_initial_empirical_objective_incumbent,
    verify_frozen_shortlist,
)
from performance.benchmark_lodo_meta_prior import build_scalarized_problem  # noqa: E402
from performance.benchmark_quality import json_safe, parse_weights  # noqa: E402


FORMAL_SOURCE_STEPS = {
    "safe_fpacoh_cbo": 10_000,
    "fsbo_cbo": 50_000,
    "malibo_cbo": 2_048,
    "metabo_cbo": 10_000,
    "hyperbo_cbo": 10_000,
    "rgpe_cbo": 1,
    "stacked_transfer_gp_cbo": 1,
    "mtgp_cbo": 1,
}


def load_source_initial_design(path, *, archive, heldout, seed, n0, dimension):
    if not path:
        raise ValueError(
            "source_informed comparison requires --initial-design-file")
    points, contract = load_frozen_source_informed_design(
        path,
        heldout=heldout,
        seed=seed,
        n0=n0,
        dimension=dimension,
    )
    if contract["source_archive_fingerprint"] != archive.fingerprint:
        raise ValueError("source-informed design archive fingerprint mismatch")
    return points, contract["fingerprint"]


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _true_metrics(problem, point):
    point = tuple(int(value) for value in point)
    objective = float(problem.true_objective(point))
    constraint_mean = float(problem.true_constraint_mean(point))
    constraint_sigma = float(problem.true_sigma(point)[1])
    chance_margin = (
        constraint_mean
        + float(norm.ppf(1.0 - float(problem.alpha))) * constraint_sigma
        - float(problem.tau)
    )
    _, optimum = problem.true_best_feasible()
    feasible = bool(chance_margin <= 0.0)
    return {
        "x_recommended": list(point),
        "true_objective": objective,
        "true_constraint_mean": constraint_mean,
        "true_constraint_sigma": constraint_sigma,
        "true_chance_margin": float(chance_margin),
        "true_feasible": feasible,
        "feasible_regret": (
            max(0.0, objective - float(optimum)) if feasible else None
        ),
        "constraint_violation": max(0.0, float(chance_margin)),
    }


def _apply_terminal_verification(problem, result, args):
    shortlist = result.get("frozen_terminal_shortlist")
    if not shortlist:
        raise RuntimeError(
            "terminal verification requires a pre-truth frozen shortlist")
    search_calls = int(result["n_simulations"])
    incumbent_audit = None
    optimizer_history = result.get("history") or []
    if bool(getattr(args, "terminal_objective_incumbent_guard", False)):
        initial_history = optimizer_history[: int(args.n0)]
        incumbent = select_initial_empirical_objective_incumbent(
            [row["x"] for row in initial_history],
            [row["observation"] for row in initial_history],
            n0=int(args.n0),
        )
        shortlist, incumbent_audit = freeze_objective_incumbent_shortlist(
            shortlist,
            incumbent,
            shortlist_size=len(shortlist),
        )
    primary_metrics = {
        key: result.get(key)
        for key in (
            "x_recommended",
            "posterior",
            "true_objective",
            "true_constraint_mean",
            "true_constraint_sigma",
            "true_chance_margin",
            "true_feasible",
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
            "transfer terminal shortlist and candidate budgets differ")
    deployed, verification = verify_frozen_shortlist(
        problem,
        shortlist,
        seed=int(args.seed),
        search_evaluation_count=search_calls,
        candidate_budgets=candidate_budgets,
        familywise_delta=float(args.terminal_verification_delta),
        method=str(args.terminal_verification_method),
        mean_delta_fraction=float(
            args.terminal_verification_mean_delta_fraction),
        shortlist_mode=str(getattr(
            args,
            "terminal_verification_shortlist_mode",
            "posterior_primary_safe_interior",
        )),
        objective_incumbent_position=(
            None
            if incumbent_audit is None
            else incumbent_audit["objective_incumbent_position"]
        ),
        objective_comparison_budget=int(getattr(
            args, "terminal_objective_comparison_budget", 0)),
        objective_comparison_delta=float(getattr(
            args, "terminal_objective_comparison_delta", 0.05)),
    )
    truth_rows = []
    for record in shortlist:
        metrics = _true_metrics(problem, record["point"])
        truth_rows.append({
            "shortlist_position": int(record["shortlist_position"]),
            "shortlist_role": str(record["shortlist_role"]),
            **metrics,
        })
    deployed_metrics = _true_metrics(problem, deployed)
    verification_calls = int(verification["verification_budget"])
    result.update(deployed_metrics)
    result.update({
        "search_recommendation": primary_metrics,
        "terminal_verification": verification,
        "frozen_terminal_shortlist": shortlist,
        "terminal_objective_incumbent": incumbent_audit,
        "terminal_verification_truth_audit": {
            "computed_after_shortlist_freeze_and_verification": True,
            "used_for_selection_or_certification": False,
            "rows": truth_rows,
        },
        "n_search_simulations": search_calls,
        "n_verification_simulations": verification_calls,
        "n_safety_verification_simulations": int(
            verification.get("safety_verification_budget", verification_calls)
        ),
        "n_objective_comparison_simulations": int(
            verification.get("objective_comparison_budget", 0)),
        "n_target_simulations_total": search_calls + verification_calls,
        "n_simulations": search_calls + verification_calls,
    })
    initial_audit = result.get("initial_truth_audit")
    if isinstance(initial_audit, dict):
        initial_audit["search_final_improves_initial_best"] = (
            initial_audit.get("final_improves_initial_best"))
        initial_regret = initial_audit.get("best_true_feasible_regret")
        initial_audit["final_improves_initial_best"] = bool(
            deployed_metrics["feasible_regret"] is not None
            and initial_regret is not None
            and float(deployed_metrics["feasible_regret"])
            < float(initial_regret) - 1e-12
        )
    target_contract = result["target_information_contract"]
    target_contract.update({
        "target_search_calls": search_calls,
        "target_verification_calls": verification_calls,
        "target_safety_verification_calls": int(
            verification.get(
                "safety_verification_budget", verification_calls)
        ),
        "target_objective_comparison_calls": int(
            verification.get("objective_comparison_budget", 0)),
        "target_total_calls": search_calls + verification_calls,
        "terminal_verification_protocol": (
            "ordered_frozen_shortlist_independent_noncentral_t"),
        "terminal_verification_familywise_delta": float(
            args.terminal_verification_delta),
        "terminal_verification_primary_budget": int(
            args.terminal_verification_primary_budget),
        "terminal_verification_support_budget": int(
            args.terminal_verification_support_budget),
        "terminal_verification_candidate_budgets": list(
            candidate_budgets),
        "terminal_verification_shortlist_mode": str(
            getattr(
                args,
                "terminal_verification_shortlist_mode",
                "posterior_primary_safe_interior",
            )),
        "terminal_objective_challenger_max_violation_probability": float(
            getattr(
                args,
                (
                    "terminal_objective_challenger_"
                    "max_violation_probability"
                ),
                0.5,
            )),
        "terminal_shortlist_frozen_before_truth_metrics": True,
        "verification_observations_update_posterior": False,
        "verification_samples_reused_from_search": False,
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
        "terminal_objective_incumbent_guard": False,
        "terminal_objective_comparison_budget": 0,
        "terminal_objective_comparison_delta": 0.05,
        "terminal_objective_challenger_max_violation_probability": 0.5,
        "terminal_verification_delta": 0.05,
        "terminal_verification_mean_delta_fraction": 0.5,
        "terminal_verification_method": "normal_quantile_tolerance",
        "terminal_safe_interior_probability_slack": 0.05,
        "terminal_safe_interior_require_provider": True,
        "source_dimension_adapter": "none",
        "source_coordinate_max_frequency": 8,
        "source_coordinate_frequency_penalty": 0.10,
    }
    for name, value in terminal_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    archive = FrozenTransferArchive.load(args.archive)
    problem = build_scalarized_problem(
        args.heldout,
        args.d,
        args.L,
        args.sigma,
        args.alpha,
        parse_weights(args.weights),
    )
    source_domains = [
        name for name in args.domains.split(",")
        if name.strip() and name.strip() != args.heldout
    ]
    archive.validate(
        expected_domains=map(str.strip, source_domains),
        expected_dimension=(
            args.d
            if args.source_dimension_adapter == "none"
            else None
        ),
    )
    if args.require_source_domains > 0 and len(archive.tasks) != int(
        args.require_source_domains
    ):
        raise ValueError("formal LODO row has the wrong number of source domains")
    initial_points = None
    initial_design_fingerprint = None
    if args.initial_design == "source_informed":
        initial_points, initial_design_fingerprint = load_source_initial_design(
            args.initial_design_file,
            archive=archive,
            heldout=args.heldout,
            seed=args.seed,
            n0=args.n0,
            dimension=problem.d,
        )
    source_steps = int(args.source_train_steps)
    if source_steps <= 0:
        source_steps = FORMAL_SOURCE_STEPS[args.method]
    checkpoint = (
        Path(args.checkpoint_dir)
        / f"seed{int(args.seed):04d}.pkl"
    )
    config = TransferBOConfig(
        method=args.method,
        N=args.N,
        n0=args.n0,
        seed=args.seed,
        candidate_pool_size=args.candidate_pool_size,
        beta_g=args.beta_g,
        beta_risk=args.beta_risk,
        initial_design=args.initial_design,
        initial_points=initial_points,
        implementation=args.implementation,
        source_train_steps=source_steps,
        target_finetune_steps=args.target_finetune_steps,
        checkpoint_path=str(checkpoint),
        checkpoint_resume=True,
        progress_logging=True,
        progress_label=(
            f"transfer:{args.implementation}:{args.heldout}:"
            f"{args.method}:seed={int(args.seed)}"
        ),
        source_dimension_adapter=args.source_dimension_adapter,
        source_coordinate_max_frequency=(
            args.source_coordinate_max_frequency),
        source_coordinate_frequency_penalty=(
            args.source_coordinate_frequency_penalty),
    )
    started = time.time()
    payload = {
        "schema_version": 2,
        "status": "running",
        "method": args.method,
        "implementation": args.implementation,
        "heldout_target_domain": args.heldout,
        "source_domains": list(archive.source_domains),
        "seed": int(args.seed),
        "comparison_contract": {
            "split": "leave_one_domain_out",
            "n_source_domains": int(len(archive.tasks)),
            "source_archive_fingerprint": archive.fingerprint,
            "source_profiles_per_domain": archive.profiles_per_domain,
            "source_simulator_calls": int(archive.simulator_calls),
            "target_dimension": int(args.d),
            "source_dimension": int(archive.tasks[0].X.shape[1]),
            "source_dimension_adapter": str(
                args.source_dimension_adapter),
            "source_coordinate_max_frequency": int(
                args.source_coordinate_max_frequency),
            "source_coordinate_frequency_penalty": float(
                args.source_coordinate_frequency_penalty),
            "dimension_adapter_uses_target_labels": False,
            "dimension_adapter_uses_target_oracle": False,
            "target_initial_calls_n0": int(args.n0),
            "target_total_calls_N": int(args.N),
            "total_source_plus_target_calls": int(
                archive.simulator_calls + args.N),
            "source_archive_identical_across_methods": True,
            "source_archive_identical_across_target_seeds": True,
            "target_initial_design": args.initial_design,
            "common_target_initial_design": (
                "frozen_source_informed_rank_spanning"
                if args.initial_design == "source_informed"
                else "scrambled_sobol"
            ),
            "source_informed_initial_proposal": bool(
                args.initial_design == "source_informed"),
            "source_informed_initial_fingerprint": (
                initial_design_fingerprint),
            "source_oracle_aided": False,
            "target_oracle_used_for_selection": False,
            "source_training_schedule": int(source_steps),
            "target_finetune_steps_per_refit": int(
                args.target_finetune_steps),
        },
    }
    try:
        optimizer = TransferConstrainedBO(problem, archive, config)
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
            result = _apply_terminal_verification(problem, result, args)
        payload["comparison_contract"][
            "target_initial_design_fingerprint"
        ] = result["target_information_contract"][
            "initial_design_fingerprint"
        ]
        if (
            initial_design_fingerprint is not None
            and payload["comparison_contract"][
                "target_initial_design_fingerprint"
            ] != initial_design_fingerprint
        ):
            raise RuntimeError(
                "optimizer did not consume the frozen initial design exactly")
        search_calls = int(result.get(
            "n_search_simulations", result["n_simulations"]))
        verification_calls = int(result.get(
            "n_verification_simulations", 0))
        target_total_calls = search_calls + verification_calls
        payload["comparison_contract"].update({
            "target_search_calls": search_calls,
            "target_verification_calls": verification_calls,
            "target_total_calls": target_total_calls,
            "total_source_plus_target_search_calls": int(
                archive.simulator_calls + search_calls),
            "total_source_plus_target_verification_calls": int(
                archive.simulator_calls + target_total_calls),
            "terminal_verification_enabled": bool(
                args.terminal_verification),
            "terminal_verification_identical_across_methods": True,
            "terminal_verification_updates_optimizer": False,
            "terminal_verification_target_oracle_used": False,
        })
        payload.update({
            "status": "ok",
            "result": result,
            "wall_time_sec": float(time.time() - started),
        })
    except Exception as exc:
        payload.update({
            "status": (
                "failed_official_runtime"
                if args.implementation == "official" else "failed"
            ),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "failure_traceback": traceback.format_exc(),
            "wall_time_sec": float(time.time() - started),
            "checkpoint_path": str(checkpoint),
        })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=TRANSFER_METHODS, required=True)
    parser.add_argument(
        "--implementation", choices=("official", "paper_core"),
        default="official",
    )
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--domains", default=(
        "FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl"
    ))
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--initial-design",
        choices=("common_sobol", "source_informed"),
        default="common_sobol",
    )
    parser.add_argument("--initial-design-file", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--require-source-domains", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=1024)
    parser.add_argument("--beta-g", type=float, default=2.0)
    parser.add_argument("--beta-risk", type=float, default=2.0)
    parser.add_argument("--source-train-steps", type=int, default=0)
    parser.add_argument("--target-finetune-steps", type=int, default=100)
    parser.add_argument(
        "--source-dimension-adapter",
        choices=("none", "ordered_dct_quadratic"),
        default="none",
    )
    parser.add_argument(
        "--source-coordinate-max-frequency", type=int, default=8)
    parser.add_argument(
        "--source-coordinate-frequency-penalty", type=float, default=0.10)
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
        "--terminal-objective-incumbent-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--terminal-objective-comparison-budget", type=int, default=0)
    parser.add_argument(
        "--terminal-objective-comparison-delta", type=float, default=0.05)
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
        "--terminal-verification-mean-delta-fraction",
        type=float,
        default=0.5,
    )
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
    print(json.dumps({
        "status": payload["status"],
        "method": args.method,
        "implementation": args.implementation,
        "heldout": args.heldout,
        "seed": args.seed,
        "out": args.out,
    }, indent=2))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
