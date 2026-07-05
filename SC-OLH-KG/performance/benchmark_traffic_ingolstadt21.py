"""Run OLH-KG / SC-OLH-KG on the live ingolstadt21 SUMO evaluator.

The output intentionally includes an original-paper-compatible `summary.json`
so the same fresh-seed validator can certify the recommendation without any
special casing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
GPR_KG_CODE = REPO_ROOT / "Final_Submission" / "GPR_KG_Code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GPR_KG_CODE))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from benchmark_quality import json_safe, parse_csv  # noqa: E402
from problems.traffic_ingolstadt21 import Ingolstadt21ScalarizedTrafficProblem  # noqa: E402


def _safe_name(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _summary_dir(args, method, partition, seed):
    out_root = Path(args.paper_results_dir)
    safe_method = _safe_name(method)
    safe_partition = _safe_name(partition)
    return out_root / f"{safe_method}_{safe_partition}_seed{int(seed)}"


def _unique_rows(rows):
    seen = set()
    out = []
    for row in rows:
        item = tuple(int(v) for v in row)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _wilson_interval(successes, n, z=1.959963984540054):
    if n <= 0:
        return float("nan"), float("nan")
    phat = float(successes) / float(n)
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    radius = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def _posterior_ranked_candidates(alg, problem, final_x, args):
    rows = [tuple(int(v) for v in final_x)]
    rows.extend(tuple(int(v) for v in x) for x in alg.observations)
    if args.final_revalidation_include_refinement:
        rows.extend(problem.recommendation_refinement_candidates())
    pool = _unique_rows(rows)
    if not pool:
        return []

    mu_obj = alg.gpr[0].posterior_mean_many(pool)
    mu_con = alg.gpr[1].posterior_mean_many(pool)
    v_con = alg.variance_model.predict_certification_variance_many(
        1, pool, problem)
    cert = alg._certification_result(mu_con, pool, v_con)
    ranked = []
    for i, x in enumerate(pool):
        margin = float(cert.margin[i])
        obs = np.asarray(alg.observations.get(x, []), dtype=float)
        observed_count = int(len(obs))
        empirical = {}
        if observed_count:
            con = obs[:, 1]
            obj = obs[:, 0]
            con_std = float(np.std(con, ddof=1)) if observed_count > 1 else 0.0
            empirical_margin = (
                float(np.mean(con))
                + float(norm.ppf(1 - problem.alpha)) * con_std
                - float(problem.tau)
            )
            empirical_p = float(np.mean(con <= problem.tau))
            empirical = {
                "empirical_mean_objective": float(np.mean(obj)),
                "empirical_mean_constraint": float(np.mean(con)),
                "empirical_std_constraint": con_std,
                "empirical_chance_margin": float(empirical_margin),
                "empirical_feasible_probability": empirical_p,
                "empirical_point_feasible": bool(empirical_p >= 1.0 - problem.alpha),
            }
        else:
            empirical = {
                "empirical_mean_objective": None,
                "empirical_mean_constraint": None,
                "empirical_std_constraint": None,
                "empirical_chance_margin": None,
                "empirical_feasible_probability": None,
                "empirical_point_feasible": False,
            }
        ranked.append({
            "x": x,
            "posterior_mu_obj": float(mu_obj[i]),
            "posterior_mu_con": float(mu_con[i]),
            "posterior_chance_margin": margin,
            "posterior_feasible": bool(margin <= 0.0),
            "observed_count": observed_count,
            **empirical,
        })
    ranked.sort(key=lambda row: (
        not row["empirical_point_feasible"],
        row["empirical_chance_margin"] if row["empirical_chance_margin"] is not None else float("inf"),
        not row["posterior_feasible"],
        row["posterior_chance_margin"],
        row["posterior_mu_obj"],
    ))
    return ranked


def _summarize_revalidation(problem, weights, x, seeds):
    t0 = time.time()
    vectors = np.array([
        problem._simulate_vector(x, seed=int(seed))
        for seed in seeds
    ], dtype=float)
    elapsed = time.time() - t0
    obj = weights[0] * vectors[:, 0] + weights[1] * vectors[:, 1]
    feasible = vectors[:, 2] <= problem.tau
    successes = int(np.sum(feasible))
    lo, hi = _wilson_interval(successes, len(seeds))
    std = vectors.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(vectors.shape[1])
    chance_margin = (
        float(np.mean(vectors[:, 2]))
        + float(norm.ppf(1 - problem.alpha)) * float(std[2])
        - float(problem.tau)
    )
    return {
        "x": [int(v) for v in x],
        "seeds": [int(v) for v in seeds],
        "R": int(len(seeds)),
        "mean_vector": [float(v) for v in vectors.mean(axis=0)],
        "std_vector": [float(v) for v in std],
        "mean_objective": float(np.mean(obj)),
        "mean_constraint": float(np.mean(vectors[:, 2])),
        "std_constraint": float(std[2]),
        "chance_margin": float(chance_margin),
        "feasible_count": successes,
        "feasible_probability": float(successes / max(1, len(seeds))),
        "wilson_95": [lo, hi],
        "point_feasible": bool(successes / max(1, len(seeds)) >= 1.0 - problem.alpha),
        "wilson_feasible": bool(lo >= 1.0 - problem.alpha),
        "wall_time_sec": float(elapsed),
    }


def _select_revalidated_candidate(rows):
    if not rows:
        return None
    wilson = [row for row in rows if row["wilson_feasible"]]
    if wilson:
        return min(wilson, key=lambda row: row["mean_objective"])
    point = [row for row in rows if row["point_feasible"]]
    if point:
        return min(point, key=lambda row: row["mean_objective"])
    return min(rows, key=lambda row: (
        -row["feasible_probability"],
        row["chance_margin"],
        row["mean_objective"],
    ))


def _final_revalidation(args, problem, alg, final, weights, seed):
    k = int(args.final_revalidation_candidates)
    r = int(args.final_revalidation_replications)
    if k <= 0 or r <= 0:
        return None
    final_x = tuple(int(v) for v in final["x_recommended"])
    ranked = _posterior_ranked_candidates(alg, problem, final_x, args)
    selected = ranked[:k]
    seed_base = int(args.final_revalidation_seed_start) + int(seed) * 100000
    seeds = [seed_base + j for j in range(r)]
    revalidated = []
    for row in selected:
        summary = _summarize_revalidation(problem, weights, row["x"], seeds)
        summary.update({
            "posterior_mu_obj": row["posterior_mu_obj"],
            "posterior_mu_con": row["posterior_mu_con"],
            "posterior_chance_margin": row["posterior_chance_margin"],
            "posterior_feasible": row["posterior_feasible"],
            "observed_count": row["observed_count"],
        })
        revalidated.append(summary)
    chosen = _select_revalidated_candidate(revalidated)
    return {
        "enabled": True,
        "candidate_count": int(k),
        "replications": int(r),
        "seed_start": int(seed_base),
        "include_refinement": bool(args.final_revalidation_include_refinement),
        "ranked_candidates": json_safe(ranked[: max(k, 10)]),
        "revalidated_candidates": json_safe(revalidated),
        "selected": json_safe(chosen),
    }


def _anchor_revalidation(args, problem, weights, seed):
    k = int(args.final_revalidation_candidates)
    r = int(args.final_revalidation_replications)
    if k <= 0 or r <= 0:
        raise ValueError("anchor_only requires positive final revalidation budget")
    candidates = problem.historical_anchor_candidates(max_count=max(k, 64))
    if args.final_revalidation_include_refinement:
        candidates.extend(problem.recommendation_refinement_candidates())
    candidates = _unique_rows(candidates)
    if not candidates:
        raise RuntimeError("anchor_only found no historical traffic anchors")
    seed_base = int(args.final_revalidation_seed_start) + int(seed) * 100000
    seeds = [seed_base + j for j in range(r)]
    revalidated = []
    for x in candidates[:k]:
        revalidated.append(_summarize_revalidation(problem, weights, x, seeds))
    chosen = _select_revalidated_candidate(revalidated)
    return {
        "enabled": True,
        "anchor_only": True,
        "candidate_count": int(k),
        "replications": int(r),
        "seed_start": int(seed_base),
        "include_refinement": bool(args.final_revalidation_include_refinement),
        "ranked_candidates": json_safe([
            {"x": list(x), "rank_source": "historical_anchor"}
            for x in candidates[: max(k, 10)]
        ]),
        "revalidated_candidates": json_safe(revalidated),
        "selected": json_safe(chosen),
    }


def _build_config(args, variant, variance_mode, seed):
    is_sc = variant.lower() in ("sc", "sc_olhkg", "olhkg_sc", "sc-olh-kg")
    return SingleOLHKGConfig(
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
        variance_mode=variance_mode,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_mean=args.lambda_mean,
        lambda_coupling=(args.lambda_coupling if is_sc else 0.0),
        beta_g=args.beta_g,
        certification_mode=args.certification_mode,
        coupling_safety_z=args.coupling_safety_z,
        coupling_gate_temperature=args.coupling_gate_temperature,
        recommendation_safety_z=args.recommendation_safety_z,
        recommendation_noise_floor_scale=args.recommendation_noise_floor_scale,
        recommendation_infeasible_penalty=args.recommendation_infeasible_penalty,
        recommendation_infeasible_strategy=args.recommendation_infeasible_strategy,
        recommendation_calibration=not args.disable_recommendation_calibration,
        recommend_observed_only=not args.allow_unobserved_recommendation,
        recommendation_axis_oracle=False,
        use_state_coupling=is_sc,
        use_state_basis=is_sc and args.use_state_basis,
        encoder_kind=args.encoder_kind,
        acquisition_mode=args.acquisition_mode,
        exact_kg_mc_samples=args.exact_kg_mc_samples,
        exact_kg_use_score=args.exact_kg_use_score,
        exact_kg_blend=args.exact_kg_blend,
        eval_pool_size=args.eval_pool_size,
        evaluate_interval=args.evaluate_interval,
        seed=seed,
    )


def _partition_name(args, variance_mode, acquisition_mode, use_state_coupling=False):
    partition = f"{variance_mode}_{acquisition_mode}"
    if use_state_coupling:
        partition += "_state"
    if args.traffic_anchor_policy != "historical":
        partition += f"_{_safe_name(args.traffic_anchor_policy)}"
    if args.tag:
        partition += f"_{_safe_name(args.tag)}"
    return partition


def run_anchor_only(args, variance_mode, seed):
    weights = [float(v) for v in parse_csv(args.weights)]
    problem = Ingolstadt21ScalarizedTrafficProblem(
        weights=weights,
        seed=seed,
        true_replications=args.true_replications,
        sigma_replications=args.sigma_replications,
        historical_anchor_policy="only",
    )
    method = "Anchor-Only"
    partition = _partition_name(args, variance_mode, "historical", use_state_coupling=False)
    out_dir = _summary_dir(args, method, partition, seed)
    summary_path = out_dir / "summary.json"
    detail_path = out_dir / "details.json"
    if args.resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    revalidation = _anchor_revalidation(args, problem, weights, seed)
    selected = revalidation["selected"]
    wall = time.time() - t0
    x_rec = [int(v) for v in selected["x"]]
    final_pareto = [x_rec]
    for row in revalidation.get("revalidated_candidates", []):
        item = list(map(int, row["x"]))
        if item not in final_pareto:
            final_pareto.append(item)
    final = {
        "x_recommended": x_rec,
        "true_objective": float(selected["mean_objective"]),
        "true_constraint_mean": float(selected["mean_constraint"]),
        "true_constraint_sigma": float(selected["std_constraint"]),
        "true_chance_margin": float(selected["chance_margin"]),
        "true_feasible": bool(selected["point_feasible"]),
        "true_vector_objectives": list(selected["mean_vector"]),
        "true_f1": float(selected["mean_vector"][0]),
        "true_f2": float(selected["mean_vector"][1]),
        "final_revalidation": revalidation,
    }
    summary = {
        "method": method,
        "partition_method": partition,
        "problem": "ingolstadt21",
        "N": 0,
        "n0": 0,
        "seed": int(seed),
        "d": int(problem.d),
        "tau": float(problem.tau),
        "alpha": float(problem.alpha),
        "weights": weights,
        "wall_time_sec": float(wall),
        "final_pareto_set": final_pareto,
        "final_recommendation": x_rec,
        "final_log": json_safe(final),
        "final_revalidation": json_safe(revalidation),
        "config": {
            "traffic_anchor_policy": "only",
            "final_revalidation_candidates": int(args.final_revalidation_candidates),
            "final_revalidation_replications": int(args.final_revalidation_replications),
            "final_revalidation_seed_start": int(args.final_revalidation_seed_start),
        },
        "traffic_note": (
            "Anchor-only ablation: no OLH-KG optimization was run; candidate "
            "selection is restricted to historical traffic anchors."
        ),
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    detail_path.write_text(json.dumps({
        "summary_path": str(summary_path),
        "final_log": json_safe(final),
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "method": method,
        "partition": partition,
        "seed": int(seed),
        "summary_path": str(summary_path),
        "x_recommended": x_rec,
        "wall_time_sec": float(wall),
    }, indent=2), flush=True)
    print("DONE", flush=True)
    return summary


def run_one(args, variant, variance_mode, seed):
    weights = [float(v) for v in parse_csv(args.weights)]
    problem = Ingolstadt21ScalarizedTrafficProblem(
        weights=weights,
        seed=seed,
        true_replications=args.true_replications,
        sigma_replications=args.sigma_replications,
        historical_anchor_policy=args.traffic_anchor_policy,
    )
    config = _build_config(args, variant, variance_mode, seed)
    method = "SC-OLH-KG" if config.use_state_coupling else "OLH-KG"
    partition = _partition_name(
        args, variance_mode, config.acquisition_mode,
        use_state_coupling=config.use_state_coupling,
    )
    out_dir = _summary_dir(args, method, partition, seed)
    summary_path = out_dir / "summary.json"
    detail_path = out_dir / "details.json"
    if args.resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    alg = SingleOLHKGAlgorithm(problem, config)
    final = alg.run(verbose=args.verbose)
    revalidation = _final_revalidation(args, problem, alg, final, weights, seed)
    if revalidation and revalidation.get("selected"):
        selected = revalidation["selected"]
        final = dict(final)
        final["pre_revalidation_x_recommended"] = list(final["x_recommended"])
        final["x_recommended"] = list(selected["x"])
        final["true_objective"] = float(selected["mean_objective"])
        final["true_constraint_mean"] = float(selected["mean_constraint"])
        final["true_constraint_sigma"] = float(selected["std_constraint"])
        final["true_chance_margin"] = float(selected["chance_margin"])
        final["true_feasible"] = bool(selected["point_feasible"])
        final["true_vector_objectives"] = list(selected["mean_vector"])
        if len(selected["mean_vector"]) >= 2:
            final["true_f1"] = float(selected["mean_vector"][0])
            final["true_f2"] = float(selected["mean_vector"][1])
        final["final_revalidation"] = revalidation
    wall = time.time() - t0
    x_rec = [int(v) for v in final["x_recommended"]]
    observed = []
    for x, ys in alg.observations.items():
        if len(observed) >= args.keep_observed_candidates:
            break
        observed.append(list(map(int, x)))
    final_pareto = [x_rec]
    if revalidation:
        for row in revalidation.get("revalidated_candidates", []):
            item = list(map(int, row["x"]))
            if item not in final_pareto:
                final_pareto.append(item)
    for row in observed:
        if row not in final_pareto:
            final_pareto.append(row)
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
        "final_revalidation": json_safe(revalidation),
        "config": json_safe(config.__dict__),
        "traffic_anchor_policy": args.traffic_anchor_policy,
        "traffic_note": (
            "Optimization uses live SUMO samples; paper-grade feasibility must "
            "be certified by validate_oos_feasibility with fresh seeds."
        ),
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    detail = {
        "summary_path": str(summary_path),
        "pre_sampling_log": json_safe(alg.pre_sampling_log),
        "iteration_log": json_safe(alg.iteration_log),
        "final_log": json_safe(final),
    }
    detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
    print(json.dumps({
        "method": method,
        "partition": partition,
        "seed": int(seed),
        "summary_path": str(summary_path),
        "x_recommended": x_rec,
        "wall_time_sec": float(wall),
    }, indent=2), flush=True)
    print("DONE", flush=True)
    return summary


def run(args):
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        for variant in parse_csv(args.variants):
            for variance_mode in parse_csv(args.variance_modes):
                if variant.lower() in {"anchor_only", "anchor-only", "anchors"}:
                    rows.append(run_anchor_only(args, variance_mode, seed))
                else:
                    rows.append(run_one(args, variant, variance_mode, seed))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="olhkg,sc_olhkg")
    parser.add_argument("--variance_modes", default="factor")
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior_pool_size", type=int, default=200)
    parser.add_argument("--posterior_keep", type=int, default=10)
    parser.add_argument("--axis_candidate_count", type=int, default=0)
    parser.add_argument("--structured_candidate_count", type=int, default=12)
    parser.add_argument("--state_candidate_count", type=int, default=20)
    parser.add_argument("--state_inverse_pool_size", type=int, default=300)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_coupling", type=float, default=0.10)
    parser.add_argument("--beta_g", type=float, default=2.0)
    parser.add_argument("--certification_mode", default="theory", choices=["theory", "legacy"])
    parser.add_argument("--coupling_safety_z", type=float, default=0.5)
    parser.add_argument("--coupling_gate_temperature", type=float, default=0.25)
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--recommendation_infeasible_strategy", default="min_margin")
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--allow_unobserved_recommendation", action="store_true")
    parser.add_argument("--use_state_basis", action="store_true")
    parser.add_argument("--encoder_kind", default="synthetic")
    parser.add_argument("--acquisition_mode", default="additive")
    parser.add_argument("--exact_kg_mc_samples", type=int, default=0)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument("--eval_pool_size", type=int, default=128)
    parser.add_argument("--evaluate_interval", type=int, default=0)
    parser.add_argument("--true_replications", type=int, default=2)
    parser.add_argument("--sigma_replications", type=int, default=3)
    parser.add_argument(
        "--traffic_anchor_policy",
        default="historical",
        choices=["historical", "none", "strict_none", "only"],
        help=(
            "historical=current behavior, none=disable old-result anchors, "
            "strict_none=also disable deterministic refinement/structured "
            "traffic shortcuts, only=anchor-only ablation"
        ),
    )
    parser.add_argument("--keep_observed_candidates", type=int, default=0)
    parser.add_argument("--final_revalidation_candidates", type=int, default=0)
    parser.add_argument("--final_revalidation_replications", type=int, default=0)
    parser.add_argument("--final_revalidation_seed_start", type=int, default=70000)
    parser.add_argument("--final_revalidation_include_refinement", action="store_true")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--paper_results_dir", default=str(GPR_KG_CODE / "results" / "ingolstadt21"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
