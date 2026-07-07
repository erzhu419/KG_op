"""Parallel LODO benchmark for learned admissible SC/HVD meta-priors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from core.admissibility import domain_tuned_audit  # noqa: E402
from performance.benchmark_quality import json_safe, parse_csv, parse_weights, write_csv  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    AdmissibleProblemAdapter,
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)


def finite_stats(values):
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def mean_bool(values):
    vals = [bool(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else None


def build_scalarized_problem(name, d, L, sigma, alpha, weights):
    return ScalarizedProblem(
        make_problem(name, d=d, L=L, sigma=sigma, alpha=alpha),
        weights=weights,
    )


def _source_augmented_problem_specs(args_dict, source_names, seed):
    rng = np.random.default_rng(int(args_dict["meta_seed"]) + 4243 * int(seed))
    n_aug = max(1, int(args_dict.get("meta_source_augments", 1)))
    base_sigma = float(args_dict["sigma"])
    base_alpha = float(args_dict["alpha"])
    base_weights = parse_weights(args_dict["weights"])
    sigma_jitter = max(float(args_dict.get("meta_source_sigma_jitter", 0.0)), 0.0)
    alpha_jitter = max(float(args_dict.get("meta_source_alpha_jitter", 0.0)), 0.0)
    weight_jitter = max(float(args_dict.get("meta_source_weight_jitter", 0.0)), 0.0)
    specs = []
    for name in source_names:
        for aug_idx in range(n_aug):
            if aug_idx == 0:
                sigma = base_sigma
                alpha = base_alpha
                weights = base_weights
                label = name
            else:
                sigma = base_sigma * float(np.exp(rng.normal(0.0, sigma_jitter)))
                alpha = base_alpha * float(np.exp(rng.normal(0.0, alpha_jitter)))
                alpha = float(np.clip(alpha, 0.01, 0.20))
                if weight_jitter > 0.0:
                    weights = np.maximum(
                        base_weights + rng.normal(0.0, weight_jitter, size=len(base_weights)),
                        1e-4,
                    )
                    weights = weights / max(float(np.sum(weights)), 1e-12)
                else:
                    weights = base_weights
                label = f"{name}#aug{aug_idx}"
            specs.append((label, name, sigma, alpha, weights))
    return specs


def train_meta_prior(args_dict, heldout, seed, *, teacher=False):
    domains = parse_csv(args_dict["domains"])
    source_names = [name for name in domains if name != heldout]
    if not source_names:
        raise ValueError(f"heldout={heldout} leaves no source domains")
    source_problems = [
        (
            label,
            build_scalarized_problem(
                name,
                args_dict["d"],
                args_dict["L"],
                sigma,
                alpha,
                weights,
            ),
        )
        for label, name, sigma, alpha, weights in _source_augmented_problem_specs(
            args_dict,
            source_names,
            seed,
        )
    ]
    prior = LearnedMetaPrior(
        local_dim=args_dict["meta_local_dim"],
        shared_dim=args_dict["meta_shared_dim"],
        anchor_count=args_dict["meta_anchor_count"],
        kmeans_iters=args_dict["meta_kmeans_iters"],
        soft_temperature=args_dict["meta_soft_temperature"],
        ridge=args_dict["meta_ridge"],
        boundary_weight=args_dict["meta_boundary_weight"],
        boundary_temperature=args_dict["meta_boundary_temperature"],
        variance_weight=args_dict["meta_variance_weight"],
        feasible_penalty=args_dict["meta_feasible_penalty"],
        feasible_bonus=args_dict["meta_feasible_bonus"],
        elite_fraction=args_dict["meta_elite_fraction"],
        boundary_fraction=args_dict["meta_boundary_fraction"],
        teacher_records_per_domain=(
            args_dict["meta_teacher_records_per_domain"] if teacher else 0
        ),
        teacher_weight=args_dict["meta_teacher_weight"],
        teacher_pool_size=args_dict["meta_teacher_pool_size"],
        teacher_elite_fraction=args_dict["meta_teacher_elite_fraction"],
        teacher_boundary_fraction=args_dict["meta_teacher_boundary_fraction"],
        anchor_sampling_temperature=(
            args_dict["meta_teacher_anchor_sampling_temperature"]
            if teacher
            else args_dict["meta_anchor_sampling_temperature"]
        ),
        hvd_noise_floor_scale=(
            args_dict["meta_teacher_hvd_noise_floor_scale"]
            if teacher
            else args_dict["meta_hvd_noise_floor_scale"]
        ),
        universal_shape_count=args_dict["meta_universal_shape_count"],
        seed=int(args_dict["meta_seed"]) + int(seed),
    )
    prior.fit_from_source_problems(
        source_problems,
        n_records_per_domain=args_dict["source_records_per_domain"],
        rng=np.random.default_rng(int(args_dict["meta_seed"]) + 1009 * int(seed)),
    )
    return prior


def build_target_problem(args_dict, heldout, line, seed):
    weights = parse_weights(args_dict["weights"])
    target = build_scalarized_problem(
        heldout,
        args_dict["d"],
        args_dict["L"],
        args_dict["sigma"],
        args_dict["alpha"],
        weights,
    )
    line = str(line)
    meta_diag = None
    if line == "strict":
        return AdmissibleProblemAdapter(target, variant="strict_universal"), meta_diag
    if line in ("lodo", "lodo_teacher"):
        prior = train_meta_prior(
            args_dict,
            heldout,
            seed,
            teacher=(line == "lodo_teacher"),
        )
        meta_diag = prior.diagnostics()
        return MetaPriorProblemAdapter(
            target,
            prior,
            proposal_pool_size=args_dict["meta_proposal_pool_size"],
            refinement_count=args_dict["meta_refinement_count"],
        ), meta_diag
    if line == "domain":
        return target, meta_diag
    raise ValueError(f"unknown line {line!r}")


def run_one(task):
    args_dict = task["args"]
    heldout = task["heldout"]
    line = task["line"]
    seed = int(task["seed"])
    basis_label = str(args_dict.get(
        "basis_label",
        "state" if bool(args_dict.get("use_state_basis", True)) else "raw",
    ))
    basis_grid = bool(args_dict.get("basis_grid", False)) or bool(
        args_dict.get("basis_pair_grid", False))
    problem, meta_diag = build_target_problem(args_dict, heldout, line, seed)
    checkpoint_path = str(args_dict.get("checkpoint_path") or "").strip()
    runtime_checkpoint_dir = str(args_dict.get("runtime_checkpoint_dir") or "").strip()
    if not runtime_checkpoint_dir and checkpoint_path:
        safe_variant = f"{line}_{heldout}_{basis_label}_seed{int(seed)}"
        safe_variant = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in safe_variant
        )
        runtime_checkpoint_dir = str(
            Path(checkpoint_path).with_suffix("").parent
            / "runtime"
            / safe_variant
        )
    if line == "strict":
        use_problem_initial = False
        use_boundary_initial = False
        use_recommendation_refinement = False
        state_candidate_count = 0
    else:
        use_problem_initial = True
        use_boundary_initial = False
        use_recommendation_refinement = True
        state_candidate_count = int(args_dict["state_candidate_count"])

    config = SingleOLHKGConfig(
        N=args_dict["N"],
        n0=args_dict["n0"],
        K1=args_dict["K1"],
        K2=args_dict["K2"],
        posterior_pool_size=args_dict["posterior_pool_size"],
        posterior_keep=args_dict["posterior_keep"],
        axis_candidate_count=args_dict["axis_candidate_count"],
        structured_candidate_count=args_dict["structured_candidate_count"],
        state_candidate_count=state_candidate_count,
        state_inverse_pool_size=args_dict["state_inverse_pool_size"],
        state_inverse_neighbors=args_dict["state_inverse_neighbors"],
        n_thr=args_dict["n_thr"],
        variance_mode=args_dict["variance_mode"],
        lambda_feas=args_dict["lambda_feas"],
        lambda_var=args_dict["lambda_var"],
        lambda_mean=args_dict["lambda_mean"],
        lambda_constraint_epistemic=args_dict["lambda_constraint_epistemic"],
        lambda_coupling=args_dict["lambda_coupling"] if line != "strict" else 0.0,
        beta_g=args_dict["beta_g"],
        certification_mode=args_dict["certification_mode"],
        recommendation_calibration=bool(args_dict["recommendation_calibration"]),
        certification_calibration=bool(args_dict["certification_calibration"]),
        recommendation_axis_oracle=False,
        use_problem_initial_samples=use_problem_initial,
        use_boundary_initial_samples=use_boundary_initial,
        use_recommendation_refinement=use_recommendation_refinement,
        use_state_coupling=True,
        use_state_basis=bool(args_dict["use_state_basis"]),
        state_basis_mode=args_dict["state_basis_mode"],
        constraint_state_basis_mode=args_dict["constraint_state_basis_mode"],
        raw_basis_dim=args_dict["raw_basis_dim"],
        raw_projection_seed=args_dict["raw_projection_seed"],
        numeric_backend=args_dict["numeric_backend"],
        numeric_backend_device=args_dict["numeric_backend_device"],
        torch_dtype=args_dict["torch_dtype"],
        torch_min_rows=args_dict["torch_min_rows"],
        encoder_kind="synthetic",
        acquisition_mode=args_dict["acquisition_mode"],
        exact_kg_mc_samples=args_dict["exact_kg_mc_samples"],
        exact_kg_jobs=args_dict["exact_kg_jobs"],
        exact_kg_use_score=args_dict["exact_kg_use_score"],
        exact_kg_blend=args_dict["exact_kg_blend"],
        constraint_uncertain_candidate_count=args_dict[
            "constraint_uncertain_candidate_count"],
        constraint_uncertain_pool_size=args_dict["constraint_uncertain_pool_size"],
        constraint_uncertain_state_pool_fraction=args_dict[
            "constraint_uncertain_state_pool_fraction"],
        constraint_uncertain_use_calibration=bool(
            args_dict["constraint_uncertain_use_calibration"]),
        constraint_epistemic_margin_softening=args_dict[
            "constraint_epistemic_margin_softening"],
        safe_interior_candidate_count=args_dict["safe_interior_candidate_count"],
        safe_interior_pool_size=args_dict["safe_interior_pool_size"],
        safe_interior_margin=args_dict["safe_interior_margin"],
        observed_neighbor_candidate_count=args_dict[
            "observed_neighbor_candidate_count"],
        observed_neighbor_radius=args_dict["observed_neighbor_radius"],
        observed_neighbor_safe_margin_scale=args_dict[
            "observed_neighbor_safe_margin_scale"],
        recommendation_infeasible_penalty=args_dict[
            "recommendation_infeasible_penalty"],
        recommendation_infeasible_strategy=args_dict[
            "recommendation_infeasible_strategy"],
        recommend_observed_only=bool(args_dict["recommend_observed_only"]),
        recommendation_slack_initial=args_dict["recommendation_slack_initial"],
        recommendation_slack_decay=args_dict["recommendation_slack_decay"],
        recommendation_calibration_scope=args_dict["recommendation_calibration_scope"],
        recommendation_calibration_min_obs=args_dict["recommendation_calibration_min_obs"],
        recommendation_calibration_max_leverage=args_dict[
            "recommendation_calibration_max_leverage"],
        recommendation_calibration_max_theory_margin=args_dict[
            "recommendation_calibration_max_theory_margin"],
        certification_calibration_min_obs=args_dict["certification_calibration_min_obs"],
        certification_calibration_beta=args_dict["certification_calibration_beta"],
        calibration_standardize_features=bool(
            args_dict["calibration_standardize_features"]),
        recommendation_observed_fallback=bool(args_dict["recommendation_observed_fallback"]),
        observed_incumbent_margin_scale=args_dict["observed_incumbent_margin_scale"],
        use_source_recommendation_slack=bool(args_dict["use_source_recommendation_slack"]),
        source_mean_prior_fallback=bool(args_dict["source_mean_prior_fallback"]),
        source_mean_prior_z=args_dict["source_mean_prior_z"],
        source_mean_prior_margin_tol=args_dict["source_mean_prior_margin_tol"],
        truth_pool_diagnostics=bool(args_dict["truth_pool_diagnostics"]),
        truth_pool_good_regret=args_dict["truth_pool_good_regret"],
        truth_pool_max_candidates=args_dict["truth_pool_max_candidates"],
        llm_prior_enabled=bool(args_dict["llm_prior_enabled"]),
        llm_prior_base_url=args_dict["llm_prior_base_url"],
        llm_prior_model=args_dict["llm_prior_model"],
        llm_prior_api_key_env=args_dict["llm_prior_api_key_env"],
        llm_prior_candidate_count=args_dict["llm_prior_candidate_count"],
        llm_prior_inverse_pool_size=args_dict["llm_prior_inverse_pool_size"],
        llm_prior_interval=args_dict["llm_prior_interval"],
        llm_prior_min_obs=args_dict["llm_prior_min_obs"],
        llm_prior_timeout_sec=args_dict["llm_prior_timeout_sec"],
        llm_prior_gate_floor=args_dict["llm_prior_gate_floor"],
        llm_prior_max_observations=args_dict["llm_prior_max_observations"],
        checkpoint_dir=runtime_checkpoint_dir,
        checkpoint_resume=bool(args_dict["runtime_checkpoint_resume"]),
        checkpoint_interval=args_dict["runtime_checkpoint_interval"],
        progress_logging=bool(args_dict["progress_logging"]),
        progress_label=(
            f"{line}:{heldout}:{basis_label}:seed{int(seed)}"
            if bool(args_dict["progress_logging"])
            else ""
        ),
        progress_units_per_iteration=args_dict["progress_units_per_iteration"],
        progress_exact_updates=args_dict["progress_exact_updates"],
        eval_pool_size=args_dict["eval_pool_size"],
        seed=seed,
    )
    started = time.time()
    alg = SingleOLHKGAlgorithm(problem, config)
    result = alg.run(verbose=False)
    true_feasible = bool(result["true_feasible"])
    posterior_feasible = bool(result.get("posterior_feasible", False))
    audit = (
        problem.admissibility_audit()
        if hasattr(problem, "admissibility_audit")
        else domain_tuned_audit().to_dict()
    )
    return {
        "line": line,
        "heldout": heldout,
        "seed": seed,
        "basis_label": basis_label,
        "state_basis_enabled": bool(args_dict["use_state_basis"]),
        "state_basis_mode": args_dict["state_basis_mode"],
        "constraint_state_basis_mode": args_dict["constraint_state_basis_mode"],
        "transfer_cell": f"{line}-{basis_label}",
        "variant": (
            f"{line}-{basis_label}:{heldout}"
            if basis_grid
            else f"{line}:{heldout}"
        ),
        "audit_admissible_mainline": bool(audit.get("admissible_mainline", False)),
        "audit": audit,
        "meta_prior": meta_diag,
        "true_feasible": true_feasible,
        "posterior_feasible": posterior_feasible,
        "false_feasible": bool(posterior_feasible and not true_feasible),
        "true_objective": float(result["true_objective"]),
        "true_best_objective": float(result["true_best_objective"]),
        "simple_regret": float(result["simple_regret"]),
        "feasible_simple_regret": (
            float(result["simple_regret"]) if true_feasible else None
        ),
        "true_chance_margin": float(result["true_chance_margin"]),
        "constraint_violation": float(max(result["true_chance_margin"], 0.0)),
        "posterior_chance_margin": float(result["posterior_chance_margin"]),
        "posterior_theory_chance_margin": result.get("posterior_theory_chance_margin"),
        "posterior_calibrated_chance_margin": result.get(
            "posterior_calibrated_chance_margin"),
        "posterior_certification_source": result.get("posterior_certification_source"),
        "recommendation_slack": result.get("recommendation_slack"),
        "recommend_observed_only": bool(args_dict["recommend_observed_only"]),
        "observed_neighbor_candidate_count": int(
            args_dict["observed_neighbor_candidate_count"]),
        "observed_neighbor_radius": float(args_dict["observed_neighbor_radius"]),
        "observed_neighbor_safe_margin_scale": float(
            args_dict["observed_neighbor_safe_margin_scale"]),
        "recommendation_calibration": result.get("recommendation_calibration"),
        "calibrated_recommendation_used": result.get("calibrated_recommendation_used"),
        "calibrated_recommendation_reason": result.get(
            "calibrated_recommendation_reason"),
        "calibrated_recommendation_scope": result.get(
            "calibrated_recommendation_scope"),
        "calibrated_recommendation_rejected_by_observed": result.get(
            "calibrated_recommendation_rejected_by_observed"),
        "calibrated_recommendation_rejected_candidate_margin": result.get(
            "calibrated_recommendation_rejected_candidate_margin"),
        "calibrated_recommendation_rejected_candidate_mu_obj": result.get(
            "calibrated_recommendation_rejected_candidate_mu_obj"),
        "n_calibration_candidates": result.get("n_calibration_candidates"),
        "n_calibration_certified_guarded": result.get(
            "n_calibration_certified_guarded"),
        "n_calibration_guarded": result.get("n_calibration_guarded"),
        "n_calibration_feasible": result.get("n_calibration_feasible"),
        "n_calibration_raw_feasible": result.get("n_calibration_raw_feasible"),
        "calibrated_constraint_margin": result.get(
            "calibrated_constraint_margin"),
        "calibrated_guarded_constraint_margin": result.get(
            "calibrated_guarded_constraint_margin"),
        "calibration_max_leverage": result.get("calibration_max_leverage"),
        "calibration_max_theory_margin": result.get(
            "calibration_max_theory_margin"),
        "calibration_selected_leverage": result.get(
            "calibration_selected_leverage"),
        "calibration_selected_theory_margin": result.get(
            "calibration_selected_theory_margin"),
        "calibration_min_leverage": result.get("calibration_min_leverage"),
        "calibration_median_leverage": result.get("calibration_median_leverage"),
        "calibration_min_theory_margin": result.get(
            "calibration_min_theory_margin"),
        "recommendation_calibration_audit_available": result.get(
            "recommendation_calibration_audit_available"),
        "recommendation_calibration_n_feasible": result.get(
            "recommendation_calibration_n_feasible"),
        "recommendation_calibration_sigma": result.get(
            "recommendation_calibration_sigma"),
        "recommendation_selected_calibrated_rec_margin": result.get(
            "recommendation_selected_calibrated_rec_margin"),
        "recommendation_selected_calibrated_rec_objective": result.get(
            "recommendation_selected_calibrated_rec_objective"),
        "recommendation_selected_calibrated_rec_leverage": result.get(
            "recommendation_selected_calibrated_rec_leverage"),
        "certification_calibration_used": result.get("certification_calibration_used"),
        "certification_calibration_n_feasible": result.get(
            "certification_calibration_n_feasible"),
        "source_mean_prior_fallback": result.get("source_mean_prior_fallback"),
        "source_mean_prior_used": result.get("source_mean_prior_used"),
        "source_mean_prior_available": result.get("source_mean_prior_available"),
        "source_mean_prior_n_feasible": result.get("source_mean_prior_n_feasible"),
        "source_mean_prior_min_margin": result.get("source_mean_prior_min_margin"),
        "source_mean_prior_selected_margin": result.get(
            "source_mean_prior_selected_margin"),
        "source_mean_prior_guard_used": result.get("source_mean_prior_guard_used"),
        "source_mean_prior_ranker_used": result.get("source_mean_prior_ranker_used"),
        "source_mean_prior_guard_n_feasible": result.get(
            "source_mean_prior_guard_n_feasible"),
        "recommendation_has_true_feasible": result.get(
            "recommendation_has_true_feasible"),
        "recommendation_missed_true_feasible": result.get(
            "recommendation_missed_true_feasible"),
        "recommendation_has_true_safe_good": result.get(
            "recommendation_has_true_safe_good"),
        "recommendation_missed_true_safe_good": result.get(
            "recommendation_missed_true_safe_good"),
        "recommendation_selected_true_feasible": result.get(
            "recommendation_selected_true_feasible"),
        "recommendation_best_true_feasible_posterior_margin": result.get(
            "recommendation_best_true_feasible_posterior_margin"),
        "recommendation_best_true_feasible_posterior_feasible": result.get(
            "recommendation_best_true_feasible_posterior_feasible"),
        "recommendation_true_best_feasible_regret": result.get(
            "recommendation_true_best_feasible_regret"),
        "recommendation_best_true_feasible_decision_margin": result.get(
            "recommendation_best_true_feasible_decision_margin"),
        "recommendation_best_true_feasible_decision_feasible": result.get(
            "recommendation_best_true_feasible_decision_feasible"),
        "recommendation_selected_decision_margin": result.get(
            "recommendation_selected_decision_margin"),
        "recommendation_best_true_feasible_mu_con": result.get(
            "recommendation_best_true_feasible_mu_con"),
        "recommendation_best_true_feasible_epistemic_var": result.get(
            "recommendation_best_true_feasible_epistemic_var"),
        "recommendation_best_true_feasible_aleatoric_var": result.get(
            "recommendation_best_true_feasible_aleatoric_var"),
        "recommendation_best_true_feasible_theory_margin": result.get(
            "recommendation_best_true_feasible_theory_margin"),
        "recommendation_best_true_feasible_calibrated_margin": result.get(
            "recommendation_best_true_feasible_calibrated_margin"),
        "recommendation_best_true_feasible_calibrated_rec_margin": result.get(
            "recommendation_best_true_feasible_calibrated_rec_margin"),
        "recommendation_best_true_feasible_calibrated_rec_objective": result.get(
            "recommendation_best_true_feasible_calibrated_rec_objective"),
        "recommendation_best_true_feasible_calibrated_rec_leverage": result.get(
            "recommendation_best_true_feasible_calibrated_rec_leverage"),
        "recommendation_best_true_feasible_source_margin": result.get(
            "recommendation_best_true_feasible_source_margin"),
        "recommendation_best_true_feasible_certification_source": result.get(
            "recommendation_best_true_feasible_certification_source"),
        "recommendation_selected_mu_con": result.get("recommendation_selected_mu_con"),
        "recommendation_selected_epistemic_var": result.get(
            "recommendation_selected_epistemic_var"),
        "recommendation_selected_aleatoric_var": result.get(
            "recommendation_selected_aleatoric_var"),
        "recommendation_selected_theory_margin": result.get(
            "recommendation_selected_theory_margin"),
        "recommendation_selected_calibrated_margin": result.get(
            "recommendation_selected_calibrated_margin"),
        "recommendation_selected_calibrated_rec_margin": result.get(
            "recommendation_selected_calibrated_rec_margin"),
        "recommendation_selected_calibrated_rec_objective": result.get(
            "recommendation_selected_calibrated_rec_objective"),
        "recommendation_selected_calibrated_rec_leverage": result.get(
            "recommendation_selected_calibrated_rec_leverage"),
        "recommendation_selected_source_margin": result.get(
            "recommendation_selected_source_margin"),
        "recommendation_selected_certification_source": result.get(
            "recommendation_selected_certification_source"),
        "observed_incumbent_used": result.get("observed_incumbent_used"),
        "observed_incumbent_rejected": result.get("observed_incumbent_rejected"),
        "observed_incumbent_reason": result.get("observed_incumbent_reason"),
        "observed_incumbent_chance_margin": result.get(
            "observed_incumbent_chance_margin"),
        "n_simulations": int(result["n_simulations"]),
        "n_distinct_solutions": int(result["n_distinct_solutions"]),
        "n_pool": int(result.get("n_pool", 0)),
        "n_posterior_feasible": int(result.get("n_posterior_feasible", 0)),
        "wall_time_sec": float(time.time() - started),
        "algorithm_time_sec": float(result["total_time_sec"]),
        "variance_diagnostics": result.get("variance", {}),
        "candidate_source_counts": result.get("candidate_source_counts", {}),
        "truth_pool_diagnostics": result.get("truth_pool_diagnostics", {}),
        "llm_prior": result.get("llm_prior", {}),
        "x_recommended": result["x_recommended"],
    }


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    out = {}
    for variant, items in grouped.items():
        out[variant] = {
            "variant": variant,
            "line": items[0]["line"],
            "heldout": items[0]["heldout"],
            "basis_label": items[0].get("basis_label", ""),
            "state_basis_enabled": bool(items[0].get("state_basis_enabled", False)),
            "state_basis_mode": items[0].get("state_basis_mode", ""),
            "constraint_state_basis_mode": items[0].get(
                "constraint_state_basis_mode", ""),
            "transfer_cell": items[0].get("transfer_cell", items[0]["line"]),
            "n_runs": len(items),
            "audit_admissible_mainline_rate": mean_bool(
                row["audit_admissible_mainline"] for row in items),
            "true_feasible_rate": mean_bool(row["true_feasible"] for row in items),
            "posterior_feasible_rate": mean_bool(row["posterior_feasible"] for row in items),
            "false_feasible_rate": mean_bool(row["false_feasible"] for row in items),
            "simple_regret": finite_stats(row["simple_regret"] for row in items),
            "feasible_simple_regret": finite_stats(
                row["feasible_simple_regret"] for row in items),
            "constraint_violation": finite_stats(
                row["constraint_violation"] for row in items),
            "true_chance_margin": finite_stats(
                row["true_chance_margin"] for row in items),
            "wall_time_sec": finite_stats(row["wall_time_sec"] for row in items),
            "llm_prior_ok_count": finite_stats(
                (row.get("llm_prior") or {}).get("ok_count", 0) for row in items),
            "llm_prior_selected_count": finite_stats(
                (row.get("llm_prior") or {}).get("selected_count", 0) for row in items),
            "llm_prior_gate_mean": finite_stats(
                (row.get("llm_prior") or {}).get("gate_mean", 0.0) for row in items),
            "pool_has_true_feasible_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "pool_has_true_feasible_rate", None)
                for row in items),
            "pool_missed_true_feasible_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "missed_true_feasible_rate", None)
                for row in items),
            "pool_has_true_safe_good_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "pool_has_true_safe_good_rate", None)
                for row in items),
            "pool_selected_true_feasible_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "selected_true_feasible_rate", None)
                for row in items),
            "pool_best_true_feasible_posterior_margin": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "mean_best_true_feasible_posterior_margin", None)
                for row in items),
            "pool_best_true_feasible_posterior_feasible_rate": finite_stats(
                (row.get("truth_pool_diagnostics") or {}).get(
                    "best_true_feasible_posterior_feasible_rate", None)
                for row in items),
            "recommendation_has_true_feasible": finite_stats(
                row.get("recommendation_has_true_feasible", None) for row in items),
            "recommendation_missed_true_feasible": finite_stats(
                row.get("recommendation_missed_true_feasible", None) for row in items),
            "recommendation_best_true_feasible_posterior_margin": finite_stats(
                row.get("recommendation_best_true_feasible_posterior_margin", None)
                for row in items),
            "recommendation_best_true_feasible_posterior_feasible": finite_stats(
                row.get("recommendation_best_true_feasible_posterior_feasible", None)
                for row in items),
            "recommendation_best_true_feasible_decision_margin": finite_stats(
                row.get("recommendation_best_true_feasible_decision_margin", None)
                for row in items),
            "recommendation_best_true_feasible_decision_feasible": finite_stats(
                row.get("recommendation_best_true_feasible_decision_feasible", None)
                for row in items),
            "recommendation_selected_decision_margin": finite_stats(
                row.get("recommendation_selected_decision_margin", None)
                for row in items),
            "source_mean_prior_used": finite_stats(
                row.get("source_mean_prior_used", None) for row in items),
            "source_mean_prior_n_feasible": finite_stats(
                row.get("source_mean_prior_n_feasible", None) for row in items),
            "source_mean_prior_guard_used": finite_stats(
                row.get("source_mean_prior_guard_used", None) for row in items),
            "source_mean_prior_ranker_used": finite_stats(
                row.get("source_mean_prior_ranker_used", None) for row in items),
            "source_mean_prior_guard_n_feasible": finite_stats(
                row.get("source_mean_prior_guard_n_feasible", None) for row in items),
            "calibrated_recommendation_rejected_by_observed": finite_stats(
                row.get("calibrated_recommendation_rejected_by_observed", None)
                for row in items),
            "observed_incumbent_used": finite_stats(
                row.get("observed_incumbent_used", None) for row in items),
            "observed_incumbent_chance_margin": finite_stats(
                row.get("observed_incumbent_chance_margin", None) for row in items),
            "recommendation_best_true_feasible_mu_con": finite_stats(
                row.get("recommendation_best_true_feasible_mu_con", None)
                for row in items),
            "recommendation_best_true_feasible_epistemic_var": finite_stats(
                row.get("recommendation_best_true_feasible_epistemic_var", None)
                for row in items),
            "recommendation_best_true_feasible_aleatoric_var": finite_stats(
                row.get("recommendation_best_true_feasible_aleatoric_var", None)
                for row in items),
            "recommendation_best_true_feasible_theory_margin": finite_stats(
                row.get("recommendation_best_true_feasible_theory_margin", None)
                for row in items),
            "recommendation_best_true_feasible_calibrated_margin": finite_stats(
                row.get("recommendation_best_true_feasible_calibrated_margin", None)
                for row in items),
            "recommendation_best_true_feasible_calibrated_rec_margin": finite_stats(
                row.get(
                    "recommendation_best_true_feasible_calibrated_rec_margin",
                    None,
                )
                for row in items),
            "recommendation_best_true_feasible_calibrated_rec_objective": finite_stats(
                row.get(
                    "recommendation_best_true_feasible_calibrated_rec_objective",
                    None,
                )
                for row in items),
            "recommendation_best_true_feasible_calibrated_rec_leverage": finite_stats(
                row.get(
                    "recommendation_best_true_feasible_calibrated_rec_leverage",
                    None,
                )
                for row in items),
            "recommendation_best_true_feasible_source_margin": finite_stats(
                row.get("recommendation_best_true_feasible_source_margin", None)
                for row in items),
            "recommendation_selected_mu_con": finite_stats(
                row.get("recommendation_selected_mu_con", None) for row in items),
            "recommendation_selected_epistemic_var": finite_stats(
                row.get("recommendation_selected_epistemic_var", None)
                for row in items),
            "recommendation_selected_aleatoric_var": finite_stats(
                row.get("recommendation_selected_aleatoric_var", None)
                for row in items),
            "recommendation_selected_theory_margin": finite_stats(
                row.get("recommendation_selected_theory_margin", None)
                for row in items),
            "recommendation_selected_calibrated_margin": finite_stats(
                row.get("recommendation_selected_calibrated_margin", None)
                for row in items),
            "recommendation_selected_calibrated_rec_margin": finite_stats(
                row.get("recommendation_selected_calibrated_rec_margin", None)
                for row in items),
            "recommendation_selected_calibrated_rec_objective": finite_stats(
                row.get("recommendation_selected_calibrated_rec_objective", None)
                for row in items),
            "recommendation_selected_calibrated_rec_leverage": finite_stats(
                row.get("recommendation_selected_calibrated_rec_leverage", None)
                for row in items),
            "recommendation_selected_source_margin": finite_stats(
                row.get("recommendation_selected_source_margin", None)
                for row in items),
            "recommendation_calibration_n_feasible": finite_stats(
                row.get("recommendation_calibration_n_feasible", None)
                for row in items),
            "recommendation_calibration_sigma": finite_stats(
                row.get("recommendation_calibration_sigma", None) for row in items),
            "n_calibration_guarded": finite_stats(
                row.get("n_calibration_guarded", None) for row in items),
            "n_calibration_certified_guarded": finite_stats(
                row.get("n_calibration_certified_guarded", None) for row in items),
            "n_calibration_feasible": finite_stats(
                row.get("n_calibration_feasible", None) for row in items),
            "n_calibration_raw_feasible": finite_stats(
                row.get("n_calibration_raw_feasible", None) for row in items),
            "calibrated_constraint_margin": finite_stats(
                row.get("calibrated_constraint_margin", None) for row in items),
            "calibrated_guarded_constraint_margin": finite_stats(
                row.get("calibrated_guarded_constraint_margin", None)
                for row in items),
            "calibration_selected_leverage": finite_stats(
                row.get("calibration_selected_leverage", None) for row in items),
            "calibration_selected_theory_margin": finite_stats(
                row.get("calibration_selected_theory_margin", None)
                for row in items),
            "calibration_min_leverage": finite_stats(
                row.get("calibration_min_leverage", None) for row in items),
            "calibration_median_leverage": finite_stats(
                row.get("calibration_median_leverage", None) for row in items),
            "calibration_min_theory_margin": finite_stats(
                row.get("calibration_min_theory_margin", None) for row in items),
        }
    return out


def flatten_summary(summary):
    row = {
        "variant": summary["variant"],
        "line": summary["line"],
        "heldout": summary["heldout"],
        "basis_label": summary.get("basis_label", ""),
        "state_basis_enabled": summary.get("state_basis_enabled", False),
        "state_basis_mode": summary.get("state_basis_mode", ""),
        "constraint_state_basis_mode": summary.get("constraint_state_basis_mode", ""),
        "transfer_cell": summary.get("transfer_cell", summary["line"]),
        "n_runs": summary["n_runs"],
        "audit_admissible_mainline_rate": summary["audit_admissible_mainline_rate"],
        "true_feasible_rate": summary["true_feasible_rate"],
        "posterior_feasible_rate": summary["posterior_feasible_rate"],
        "false_feasible_rate": summary["false_feasible_rate"],
    }
    for metric in (
        "simple_regret",
        "feasible_simple_regret",
        "constraint_violation",
        "true_chance_margin",
        "wall_time_sec",
        "llm_prior_ok_count",
        "llm_prior_selected_count",
        "llm_prior_gate_mean",
        "pool_has_true_feasible_rate",
        "pool_missed_true_feasible_rate",
        "pool_has_true_safe_good_rate",
        "pool_selected_true_feasible_rate",
        "pool_best_true_feasible_posterior_margin",
        "pool_best_true_feasible_posterior_feasible_rate",
        "recommendation_has_true_feasible",
        "recommendation_missed_true_feasible",
        "recommendation_best_true_feasible_posterior_margin",
        "recommendation_best_true_feasible_posterior_feasible",
        "recommendation_best_true_feasible_decision_margin",
        "recommendation_best_true_feasible_decision_feasible",
        "recommendation_selected_decision_margin",
        "source_mean_prior_used",
        "source_mean_prior_n_feasible",
        "source_mean_prior_guard_used",
        "source_mean_prior_ranker_used",
        "source_mean_prior_guard_n_feasible",
        "calibrated_recommendation_rejected_by_observed",
        "observed_incumbent_used",
        "observed_incumbent_chance_margin",
        "recommendation_best_true_feasible_mu_con",
        "recommendation_best_true_feasible_epistemic_var",
        "recommendation_best_true_feasible_aleatoric_var",
        "recommendation_best_true_feasible_theory_margin",
        "recommendation_best_true_feasible_calibrated_margin",
        "recommendation_best_true_feasible_calibrated_rec_margin",
        "recommendation_best_true_feasible_calibrated_rec_objective",
        "recommendation_best_true_feasible_calibrated_rec_leverage",
        "recommendation_best_true_feasible_source_margin",
        "recommendation_selected_mu_con",
        "recommendation_selected_epistemic_var",
        "recommendation_selected_aleatoric_var",
        "recommendation_selected_theory_margin",
        "recommendation_selected_calibrated_margin",
        "recommendation_selected_calibrated_rec_margin",
        "recommendation_selected_calibrated_rec_objective",
        "recommendation_selected_calibrated_rec_leverage",
        "recommendation_selected_source_margin",
        "recommendation_calibration_n_feasible",
        "recommendation_calibration_sigma",
        "n_calibration_guarded",
        "n_calibration_certified_guarded",
        "n_calibration_feasible",
        "n_calibration_raw_feasible",
        "calibrated_constraint_margin",
        "calibrated_guarded_constraint_margin",
        "calibration_selected_leverage",
        "calibration_selected_theory_margin",
        "calibration_min_leverage",
        "calibration_median_leverage",
        "calibration_min_theory_margin",
    ):
        for key, value in summary.get(metric, {}).items():
            row[f"{metric}_{key}"] = value
    return row


def run(args):
    heldouts = parse_csv(args.heldouts) or parse_csv(args.domains)
    lines = parse_csv(args.lines)
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    args_dict = vars(args).copy()
    if args.basis_pair_grid:
        basis_options = [
            (
                "same_raw_state",
                True,
                str(args.state_basis_mode),
                "",
            ),
            (
                "split_constraint_state",
                True,
                str(args.state_basis_mode),
                "state",
            ),
        ]
    elif args.basis_grid:
        basis_options = [
            ("raw", False, str(args.state_basis_mode), str(args.constraint_state_basis_mode)),
            ("state", True, str(args.state_basis_mode), str(args.constraint_state_basis_mode)),
        ]
    else:
        basis_options = [
            (
                "state" if bool(args.use_state_basis) else "raw",
                bool(args.use_state_basis),
                str(args.state_basis_mode),
                str(args.constraint_state_basis_mode),
            )
        ]
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    resume_paths = []
    if checkpoint_path:
        resume_paths.append(checkpoint_path)
    for path in parse_csv(getattr(args, "resume_completed_from", "")):
        extra = Path(path)
        if extra not in resume_paths:
            resume_paths.append(extra)
    existing_rows = []
    completed = set()
    if args.resume_completed:
        for resume_path in resume_paths:
            if not resume_path.exists():
                continue
            for line in resume_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    str(row["heldout"]),
                    str(row["line"]),
                    int(row["seed"]),
                    str(row.get(
                        "basis_label",
                        "state" if row.get("state_basis_enabled", True) else "raw",
                    )),
                )
                if key not in completed:
                    existing_rows.append(row)
                    completed.add(key)
            print(
                f"[lodo] resumed_rows={len(existing_rows)} "
                f"from {resume_path}",
                flush=True,
            )
    tasks = []
    shard_index = int(getattr(args, "task_shard_index", 0))
    num_shards = max(1, int(getattr(args, "task_num_shards", 1)))
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"--task_shard_index must be in [0,{num_shards}); got {shard_index}"
        )
    task_ordinal = 0
    for heldout in heldouts:
        for line in lines:
            for seed in seeds:
                for (
                    basis_label,
                    use_state_basis,
                    state_basis_mode,
                    constraint_state_basis_mode,
                ) in basis_options:
                    belongs_to_shard = (task_ordinal % num_shards) == shard_index
                    task_ordinal += 1
                    if not belongs_to_shard:
                        continue
                    key = (str(heldout), str(line), int(seed), str(basis_label))
                    if key in completed:
                        continue
                    task_args = dict(args_dict)
                    task_args["basis_label"] = str(basis_label)
                    task_args["use_state_basis"] = bool(use_state_basis)
                    task_args["state_basis_mode"] = str(state_basis_mode)
                    task_args["constraint_state_basis_mode"] = str(
                        constraint_state_basis_mode)
                    tasks.append({
                        "args": task_args,
                        "heldout": heldout,
                        "line": line,
                        "seed": seed,
                    })
    rows = list(existing_rows)
    total = len(tasks)
    print(
        f"[lodo] tasks={total} resumed={len(existing_rows)} jobs={args.jobs} "
        f"shard={shard_index}/{num_shards}",
        flush=True,
    )
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def save_row(row):
        if not checkpoint_path:
            return
        with checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
            fh.flush()

    if args.jobs <= 1:
        for idx, task in enumerate(tasks):
            print(
                f"[{idx}/{total}] start line={task['line']} "
                f"heldout={task['heldout']} seed={task['seed']} "
                f"basis={task['args'].get('basis_label', '')}",
                flush=True,
            )
            row = run_one(task)
            rows.append(row)
            save_row(row)
            print(
                f"Step {idx + 1}/{total} [lodo-run] done "
                f"line={task['line']} heldout={task['heldout']} "
                f"seed={task['seed']} "
                f"basis={task['args'].get('basis_label', '')}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            done = 0
            for future in as_completed(futures):
                task = futures[future]
                row = future.result()
                rows.append(row)
                save_row(row)
                done += 1
                print(
                    f"Step {done}/{total} [lodo-run] "
                    f"[{done}/{total}] done line={task['line']} "
                    f"heldout={task['heldout']} seed={task['seed']} "
                    f"basis={task['args'].get('basis_label', '')}",
                    flush=True,
                )
    summaries = summarize(rows)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": json_safe(args_dict),
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"lodo_meta_prior_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, [flatten_summary(v) for v in result["summary"].values()])
    return {"json": str(json_path), "rows_csv": str(rows_path), "summary_csv": str(summary_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl",
    )
    parser.add_argument("--heldouts", default="")
    parser.add_argument("--lines", default="strict,lodo,domain")
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=25)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior_pool_size", type=int, default=300)
    parser.add_argument("--posterior_keep", type=int, default=15)
    parser.add_argument("--axis_candidate_count", type=int, default=-1)
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=12)
    parser.add_argument("--state_inverse_pool_size", type=int, default=300)
    parser.add_argument("--state_inverse_neighbors", type=int, default=1)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--eval_pool_size", type=int, default=300)
    parser.add_argument("--variance_mode", default="factor")
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_constraint_epistemic", type=float, default=0.0)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--beta_g", type=float, default=2.0)
    parser.add_argument("--certification_mode", default="theory")
    parser.add_argument("--use_state_basis", dest="use_state_basis", action="store_true", default=True)
    parser.add_argument("--disable_state_basis", dest="use_state_basis", action="store_false")
    parser.add_argument(
        "--basis_grid",
        action="store_true",
        help="Run raw/state basis variants for the core transfer ablation table.",
    )
    parser.add_argument(
        "--basis_pair_grid",
        action="store_true",
        help=(
            "Run same raw+state mean basis versus split objective raw+state / "
            "constraint state basis."
        ),
    )
    parser.add_argument("--state_basis_mode", default="raw+state")
    parser.add_argument("--constraint_state_basis_mode", default="")
    parser.add_argument("--raw_basis_dim", type=int, default=32)
    parser.add_argument("--raw_projection_seed", type=int, default=314159)
    parser.add_argument("--numeric_backend", default="numpy")
    parser.add_argument("--numeric_backend_device", default="auto")
    parser.add_argument("--torch_dtype", default="float64")
    parser.add_argument("--torch_min_rows", type=int, default=128)
    parser.add_argument("--acquisition_mode", default="additive")
    parser.add_argument("--exact_kg_mc_samples", type=int, default=0)
    parser.add_argument("--exact_kg_jobs", type=int, default=1)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument("--constraint_uncertain_candidate_count", type=int, default=0)
    parser.add_argument("--constraint_uncertain_pool_size", type=int, default=300)
    parser.add_argument("--constraint_uncertain_state_pool_fraction", type=float, default=0.25)
    parser.add_argument("--constraint_uncertain_use_calibration", action="store_true")
    parser.add_argument("--constraint_epistemic_margin_softening", type=float, default=3.0)
    parser.add_argument("--safe_interior_candidate_count", type=int, default=0)
    parser.add_argument("--safe_interior_pool_size", type=int, default=300)
    parser.add_argument("--safe_interior_margin", type=float, default=0.0)
    parser.add_argument("--observed_neighbor_candidate_count", type=int, default=0)
    parser.add_argument("--observed_neighbor_radius", type=float, default=0.08)
    parser.add_argument("--observed_neighbor_safe_margin_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--recommendation_infeasible_strategy", default="penalty")
    parser.add_argument("--recommend_observed_only", action="store_true")
    parser.add_argument("--recommendation_slack_initial", type=float, default=0.0)
    parser.add_argument("--recommendation_slack_decay", default="sqrt")
    parser.add_argument("--recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_scope", default="refinement")
    parser.add_argument("--recommendation_calibration_min_obs", type=int, default=8)
    parser.add_argument("--recommendation_calibration_max_leverage", type=float, default=0.0)
    parser.add_argument(
        "--recommendation_calibration_max_theory_margin",
        type=float,
        default=0.0,
    )
    parser.add_argument("--recommendation_observed_fallback", action="store_true")
    parser.add_argument("--observed_incumbent_margin_scale", type=float, default=-0.5)
    parser.add_argument("--certification_calibration", action="store_true")
    parser.add_argument("--certification_calibration_min_obs", type=int, default=8)
    parser.add_argument("--certification_calibration_beta", type=float, default=2.0)
    parser.add_argument("--calibration_standardize_features", action="store_true")
    parser.add_argument("--use_source_recommendation_slack", action="store_true")
    parser.add_argument("--source_mean_prior_fallback", action="store_true")
    parser.add_argument("--source_mean_prior_z", type=float, default=1.0)
    parser.add_argument("--source_mean_prior_margin_tol", type=float, default=0.0)
    parser.add_argument("--truth_pool_diagnostics", action="store_true")
    parser.add_argument("--truth_pool_good_regret", type=float, default=0.05)
    parser.add_argument("--truth_pool_max_candidates", type=int, default=0)
    parser.add_argument("--llm_prior_enabled", action="store_true")
    parser.add_argument("--llm_prior_base_url", default="https://ruoli.dev")
    parser.add_argument("--llm_prior_model", default="gpt-5.4-mini")
    parser.add_argument("--llm_prior_api_key_env", default="SCOLHKG_LLM_API_KEY")
    parser.add_argument("--llm_prior_candidate_count", type=int, default=8)
    parser.add_argument("--llm_prior_inverse_pool_size", type=int, default=256)
    parser.add_argument("--llm_prior_interval", type=int, default=5)
    parser.add_argument("--llm_prior_min_obs", type=int, default=8)
    parser.add_argument("--llm_prior_timeout_sec", type=float, default=30.0)
    parser.add_argument("--llm_prior_gate_floor", type=float, default=0.05)
    parser.add_argument("--llm_prior_max_observations", type=int, default=24)
    parser.add_argument("--runtime_checkpoint_dir", default="")
    parser.add_argument("--runtime_checkpoint_resume", action="store_true")
    parser.add_argument("--runtime_checkpoint_interval", type=int, default=1)
    parser.add_argument("--progress_logging", action="store_true")
    parser.add_argument("--progress_units_per_iteration", type=int, default=100)
    parser.add_argument("--progress_exact_updates", type=int, default=10)
    parser.add_argument("--source_records_per_domain", type=int, default=96)
    parser.add_argument("--meta_local_dim", type=int, default=3)
    parser.add_argument("--meta_shared_dim", type=int, default=3)
    parser.add_argument("--meta_anchor_count", type=int, default=24)
    parser.add_argument("--meta_kmeans_iters", type=int, default=25)
    parser.add_argument("--meta_soft_temperature", type=float, default=0.75)
    parser.add_argument("--meta_ridge", type=float, default=1e-4)
    parser.add_argument("--meta_boundary_weight", type=float, default=1.0)
    parser.add_argument("--meta_boundary_temperature", type=float, default=1.0)
    parser.add_argument("--meta_variance_weight", type=float, default=0.5)
    parser.add_argument("--meta_feasible_penalty", type=float, default=6.0)
    parser.add_argument("--meta_feasible_bonus", type=float, default=0.15)
    parser.add_argument("--meta_elite_fraction", type=float, default=0.40)
    parser.add_argument("--meta_boundary_fraction", type=float, default=0.35)
    parser.add_argument("--meta_anchor_sampling_temperature", type=float, default=0.0)
    parser.add_argument("--meta_teacher_records_per_domain", type=int, default=96)
    parser.add_argument("--meta_teacher_weight", type=float, default=3.0)
    parser.add_argument("--meta_teacher_pool_size", type=int, default=2048)
    parser.add_argument("--meta_teacher_elite_fraction", type=float, default=0.50)
    parser.add_argument("--meta_teacher_boundary_fraction", type=float, default=0.35)
    parser.add_argument(
        "--meta_teacher_anchor_sampling_temperature",
        type=float,
        default=0.35,
    )
    parser.add_argument("--meta_hvd_noise_floor_scale", type=float, default=0.0)
    parser.add_argument("--meta_teacher_hvd_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--meta_universal_shape_count", type=int, default=64)
    parser.add_argument("--meta_source_augments", type=int, default=1)
    parser.add_argument("--meta_source_sigma_jitter", type=float, default=0.20)
    parser.add_argument("--meta_source_alpha_jitter", type=float, default=0.25)
    parser.add_argument("--meta_source_weight_jitter", type=float, default=0.05)
    parser.add_argument("--meta_seed", type=int, default=20260706)
    parser.add_argument("--meta_proposal_pool_size", type=int, default=512)
    parser.add_argument("--meta_refinement_count", type=int, default=96)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--resume_completed_from", default="")
    parser.add_argument("--resume_completed", action="store_true")
    parser.add_argument("--task_shard_index", type=int, default=0)
    parser.add_argument("--task_num_shards", type=int, default=1)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()
    if args.N <= args.n0:
        raise ValueError("--N must be larger than --n0")
    result = run(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({"paths": paths, "summary": result["summary"]}), indent=2))


if __name__ == "__main__":
    main()
