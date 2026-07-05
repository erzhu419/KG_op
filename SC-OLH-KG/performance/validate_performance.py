"""Quality-first validation gate for SC-OLH-KG development.

The first accepted baseline is the legacy GPR-KG implementation.  After a
candidate passes, `--promote-if-pass` writes `baselines/current.json`; future
runs compare against that promoted baseline.

This gate treats optimization quality as the primary acceptance criterion.
Wall time is only an engineering constraint so experiments remain runnable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
LEGACY_ROOT = REPO_ROOT / "Final_Submission" / "GPR_KG_Code"
BASELINE_FILE = ROOT / "performance" / "baselines" / "current.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEGACY_ROOT))

from algorithms.biobj_smoke import BiObjSmokeConfig, BiObjectiveOLHKGSmoke  # noqa: E402
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


def median(values):
    return float(statistics.median(values)) if values else float("nan")


def optional_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric_delta(candidate, baseline):
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)


def nonworse_minimization(candidate, baseline, tolerance):
    if baseline is None:
        return True
    if candidate is None:
        return False
    return bool(candidate <= baseline + tolerance)


def compact_stage_times(stage_times):
    return {
        key: {
            "total": float(val.get("total", 0.0)),
            "mean": float(val.get("mean", 0.0)),
            "share": float(val.get("share", 0.0)),
        }
        for key, val in stage_times.items()
    }


def run_legacy_once(args, seed):
    from gpr_kg import GPRKR_Algorithm, RZDT1  # imported lazily from legacy root

    problem = RZDT1(
        d=args.d,
        L=100,
        sigma=args.sigma,
        heteroscedastic=True,
        alpha=args.alpha,
    )
    problem.tau = 0.0
    alg = GPRKR_Algorithm(
        problem=problem,
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        K2=args.legacy_K2,
        n_thr=args.n_thr,
        seed=seed,
        partition_method="binary_bin",
        use_boundary_initial_design=True,
    )
    t0 = time.perf_counter()
    alg.run(verbose=False)
    wall = time.perf_counter() - t0
    stage = {}
    keys = [
        "t_posterior_solve",
        "t_candidate_gen",
        "t_kg_compute",
        "t_simulate",
        "t_belief_update",
        "t_vepm_update",
        "t_hv_eval",
    ]
    denom = max(len(alg.iteration_log), 1)
    total = 0.0
    for key in keys:
        s = float(sum(row.get(key, 0.0) for row in alg.iteration_log))
        stage[key] = {"total": s, "mean": s / denom}
        total += s
    for key in keys:
        stage[key]["share"] = stage[key]["total"] / total if total > 0 else 0.0
    hv = alg.hv_history[-1][1] if alg.hv_history else None
    return {
        "wall_time_sec": float(wall),
        "stage_times": compact_stage_times(stage),
        "n_simulations": int(len(alg.history)),
        "n_distinct_solutions": int(len(alg.gpr[0].sampled_set)),
        "hv_last": None if hv is None else float(hv),
    }


def summarize_runs(name, runs):
    walls = [row["wall_time_sec"] for row in runs]
    out = {
        "name": name,
        "wall_time_sec": median(walls),
        "wall_time_sec_runs": [float(v) for v in walls],
        "runs": len(runs),
    }
    last = runs[-1]
    for key in (
        "stage_times",
        "n_simulations",
        "n_distinct_solutions",
        "hv_last",
        "true_feasible",
        "simple_regret",
        "true_objective",
        "true_best_objective",
        "hv_final",
        "pareto_size",
        "variance_mode",
    ):
        if key in last:
            out[key] = last[key]
    return out


def run_repeated(name, fn, repeats, seed_base):
    rows = []
    for rep in range(repeats):
        rows.append(fn(seed_base + rep))
    return summarize_runs(name, rows)


def run_single_once(args, seed, use_state_coupling=False):
    base = make_problem(args.candidate_problem, d=args.d, sigma=args.sigma, alpha=args.alpha)
    problem = ScalarizedProblem(base)
    config = SingleOLHKGConfig(
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        K2=args.candidate_K2,
        posterior_pool_size=args.posterior_pool_size,
        posterior_keep=args.posterior_keep,
        structured_candidate_count=args.structured_candidate_count,
        state_candidate_count=args.state_candidate_count,
        state_inverse_pool_size=args.state_inverse_pool_size,
        state_inverse_neighbors=args.state_inverse_neighbors,
        n_thr=args.n_thr,
        variance_mode=args.variance_mode,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_mean=args.lambda_mean,
        lambda_coupling=args.lambda_coupling if use_state_coupling else 0.0,
        coupling_safety_z=args.coupling_safety_z,
        coupling_gate_temperature=args.coupling_gate_temperature,
        recommendation_infeasible_penalty=args.recommendation_infeasible_penalty,
        recommendation_calibration=not args.disable_recommendation_calibration,
        recommendation_calibration_ridge=args.recommendation_calibration_ridge,
        use_state_coupling=use_state_coupling,
        use_state_basis=bool(use_state_coupling and args.use_state_basis),
        state_basis_mode=args.state_basis_mode,
        encoder_kind=args.encoder_kind,
        encoder_latent_dim=args.encoder_latent_dim,
        encoder_fit_pool_size=args.encoder_fit_pool_size,
        seed=seed,
    )
    alg = SingleOLHKGAlgorithm(problem, config)
    result = alg.run(verbose=False)
    return {
        "wall_time_sec": float(result["total_time_sec"]),
        "stage_times": compact_stage_times(result["stage_times"]),
        "n_simulations": int(result["n_simulations"]),
        "n_distinct_solutions": int(result["n_distinct_solutions"]),
        "true_feasible": bool(result["true_feasible"]),
        "simple_regret": float(result["simple_regret"]),
        "true_objective": float(result["true_objective"]),
        "true_best_objective": float(result["true_best_objective"]),
        "variance_mode": args.variance_mode,
    }


def run_biobj_once(args, seed):
    problem = make_problem(args.legacy_problem, d=args.d, sigma=args.sigma, alpha=args.alpha)
    config = BiObjSmokeConfig(
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        variance_mode=args.biobj_variance_mode,
        seed=seed,
    )
    alg = BiObjectiveOLHKGSmoke(problem, config)
    result = alg.run(verbose=False)
    return {
        "wall_time_sec": float(result["total_time_sec"]),
        "stage_times": compact_stage_times(result["stage_times"]),
        "n_simulations": int(result["n_simulations"]),
        "hv_final": float(result["hv_final"]),
        "pareto_size": int(result["pareto_size"]),
        "variance_mode": args.biobj_variance_mode,
    }


def load_baseline(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_primary_wall(baseline):
    if baseline is None:
        return None
    if "promoted_candidate" in baseline:
        return float(baseline["promoted_candidate"]["primary_wall_time_sec"])
    if "legacy_baseline" in baseline:
        return float(baseline["legacy_baseline"]["wall_time_sec"])
    return None


def baseline_primary_metrics(baseline):
    if baseline is None:
        return None
    if "promoted_candidate" in baseline:
        candidate = baseline["promoted_candidate"]
        metrics = candidate.get("metrics", {})
        return {
            "wall_time_sec": optional_float(
                candidate.get("primary_wall_time_sec", metrics.get("wall_time_sec"))
            ),
            "simple_regret": optional_float(metrics.get("simple_regret")),
            "true_objective": optional_float(metrics.get("true_objective")),
            "true_best_objective": optional_float(metrics.get("true_best_objective")),
            "true_feasible": bool(metrics.get("true_feasible", False)),
        }
    wall = baseline_primary_wall(baseline)
    if wall is not None:
        return {
            "wall_time_sec": float(wall),
            "simple_regret": None,
            "true_objective": None,
            "true_best_objective": None,
            "true_feasible": None,
        }
    return None


def build_validation(args):
    baseline_doc = load_baseline(BASELINE_FILE)
    active_baseline_kind = "promoted" if baseline_doc is not None else "legacy"

    legacy = None
    if baseline_doc is None or args.always_run_legacy:
        legacy = run_repeated(
            "legacy_gprkg",
            lambda seed: run_legacy_once(args, seed),
            args.repeats,
            args.seed,
        )

    single = run_repeated(
        "single_olhkg",
        lambda seed: run_single_once(args, seed, use_state_coupling=False),
        args.repeats,
        args.seed,
    )
    sc = run_repeated(
        "sc_olhkg",
        lambda seed: run_single_once(args, seed, use_state_coupling=True),
        args.repeats,
        args.seed,
    )
    biobj = run_repeated(
        "biobj_olhkg_smoke",
        lambda seed: run_biobj_once(args, seed),
        args.repeats,
        args.seed,
    )

    baseline_metrics = baseline_primary_metrics(baseline_doc)
    baseline_source = "promoted_candidate"
    if baseline_metrics is None:
        baseline_metrics = {
            "wall_time_sec": float(legacy["wall_time_sec"]),
            "simple_regret": None,
            "true_objective": None,
            "true_best_objective": None,
            "true_feasible": None,
        }
        baseline_source = "legacy_gprkg"

    primary = single
    baseline_wall = baseline_metrics["wall_time_sec"]
    primary_wall = optional_float(primary.get("wall_time_sec"))
    ratio = (
        float(primary_wall / baseline_wall)
        if baseline_wall and primary_wall
        else float("inf")
    )
    baseline_regret = baseline_metrics.get("simple_regret")
    candidate_regret = optional_float(primary.get("simple_regret"))
    baseline_objective = baseline_metrics.get("true_objective")
    candidate_objective = optional_float(primary.get("true_objective"))

    gates = {
        "wall_time_ratio": ratio,
        "wall_time_constraint_pass": bool(ratio <= args.max_wall_slowdown),
        "baseline_simple_regret": baseline_regret,
        "candidate_simple_regret": candidate_regret,
        "simple_regret_delta": metric_delta(candidate_regret, baseline_regret),
        "quality_regret_pass": nonworse_minimization(
            candidate_regret,
            baseline_regret,
            args.max_regret_delta,
        ),
        "baseline_true_objective": baseline_objective,
        "candidate_true_objective": candidate_objective,
        "true_objective_delta": metric_delta(candidate_objective, baseline_objective),
        "quality_objective_pass": nonworse_minimization(
            candidate_objective,
            baseline_objective,
            args.max_objective_delta,
        ),
        "single_feasible_pass": bool(single.get("true_feasible", False)),
        "sc_smoke_pass": bool(sc.get("n_simulations", 0) == args.N),
        "sc_feasible_observed": bool(sc.get("true_feasible", False)),
        "sc_feasible_required": bool(args.require_sc_feasible),
        "biobj_smoke_pass": bool(
            biobj.get("pareto_size", 0) > 0 and biobj.get("hv_final", 0.0) >= 0.0),
    }
    if args.require_sc_feasible:
        gates["sc_feasible_pass"] = bool(sc.get("true_feasible", False))
    gates["all_pass"] = bool(all(v for k, v in gates.items() if k.endswith("_pass")))

    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validation_config": {
            "N": args.N,
            "n0": args.n0,
            "K1": args.K1,
            "legacy_K2": args.legacy_K2,
            "candidate_K2": args.candidate_K2,
            "posterior_pool_size": args.posterior_pool_size,
            "posterior_keep": args.posterior_keep,
            "repeats": args.repeats,
            "seed": args.seed,
            "max_wall_slowdown": args.max_wall_slowdown,
            "max_regret_delta": args.max_regret_delta,
            "max_objective_delta": args.max_objective_delta,
            "require_sc_feasible": args.require_sc_feasible,
            "use_state_basis": args.use_state_basis,
            "state_basis_mode": args.state_basis_mode,
            "encoder_kind": args.encoder_kind,
            "variance_mode": args.variance_mode,
            "candidate_problem": args.candidate_problem,
            "legacy_problem": args.legacy_problem,
        },
        "active_baseline_kind": active_baseline_kind,
        "baseline_source": baseline_source,
        "baseline_file": str(BASELINE_FILE),
        "baseline_metrics": baseline_metrics,
        "baseline_wall_time_sec": float(baseline_wall),
        "legacy_baseline": legacy,
        "candidates": {
            "single_olhkg": single,
            "sc_olhkg": sc,
            "biobj_olhkg_smoke": biobj,
        },
        "primary_candidate": "single_olhkg",
        "primary_wall_time_sec": None if primary_wall is None else float(primary_wall),
        "gates": gates,
    }


def promote(validation):
    doc = {
        "schema_version": 1,
        "created_at": validation["created_at"],
        "baseline_policy": (
            "Quality-first baseline. Promotion requires feasible primary and "
            "state-coupled and bi-objective smoke runs, non-worse "
            "simple_regret/true_objective when those metrics exist, and wall "
            "time within max_wall_slowdown. SC feasibility is recorded but is "
            "only a hard gate when require_sc_feasible is enabled. Faster wall "
            "time alone is not an optimization improvement."
        ),
        "promoted_candidate": {
            "name": validation["primary_candidate"],
            "primary_wall_time_sec": validation["primary_wall_time_sec"],
            "metrics": validation["candidates"][validation["primary_candidate"]],
        },
        "validation_config": validation["validation_config"],
        "acceptance_gates": validation["gates"],
        "full_validation": validation,
    }
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(json_safe(doc), indent=2), encoding="utf-8")
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--n0", type=int, default=5)
    parser.add_argument("--K1", type=int, default=10)
    parser.add_argument("--legacy_K2", type=int, default=1)
    parser.add_argument("--candidate_K2", type=int, default=1)
    parser.add_argument("--posterior_pool_size", type=int, default=150)
    parser.add_argument("--posterior_keep", type=int, default=8)
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=-1)
    parser.add_argument("--state_inverse_pool_size", type=int, default=500)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--legacy_problem", default="RZDT1")
    parser.add_argument("--candidate_problem", default="RegimeRZDT1")
    parser.add_argument("--variance_mode", default="orthogonal",
                        choices=["pooled", "oracle", "class", "orthogonal", "factor"])
    parser.add_argument("--biobj_variance_mode", default="class",
                        choices=["pooled", "oracle", "class", "orthogonal", "factor"])
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--coupling_safety_z", type=float, default=0.5)
    parser.add_argument("--coupling_gate_temperature", type=float, default=0.25)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--max_wall_slowdown", "--max_wall_ratio", dest="max_wall_slowdown",
                        type=float, default=1.25)
    parser.add_argument("--max_regret_delta", type=float, default=0.0)
    parser.add_argument("--max_objective_delta", type=float, default=0.0)
    parser.add_argument("--require-sc-feasible", action="store_true")
    parser.add_argument("--use_state_basis", action="store_true")
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
            "ssl_masked",
            "ssl_contrastive",
            "ssl_next_risk",
            "ssl_transformer",
        ],
    )
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--always-run-legacy", action="store_true")
    parser.add_argument("--promote-if-pass", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    validation = build_validation(args)
    promoted = None
    if args.promote_if_pass and validation["gates"]["all_pass"]:
        promoted = promote(validation)

    output = {
        "validation": validation,
        "promoted": promoted is not None,
    }
    print(json.dumps(json_safe(output), indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(json_safe(output), indent=2), encoding="utf-8")

    if not validation["gates"]["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
