"""Offline terminal-ranking audit from a final runtime checkpoint.

The posterior, terminal pool, selected action, and comparison actions are
frozen before synthetic truth is queried.  The audit never calls the noisy
simulator and is not an admissible decision input.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    build_target_problem,
)
from performance.benchmark_quality import json_safe  # noqa: E402
from performance.diagnose_ordered_coordinate import _auc  # noqa: E402


def _load_source_result(path):
    payload = json.loads(Path(path).read_text())
    rows = list(payload.get("rows", []))
    if len(rows) != 1:
        raise ValueError("terminal audit expects one row per source shard")
    return dict(payload["config"]), rows[0]


def _restore(source_result, checkpoint_path):
    profile_config, row = _load_source_result(source_result)
    with Path(checkpoint_path).open("rb") as handle:
        checkpoint = pickle.load(handle)
    problem, _ = build_target_problem(
        profile_config,
        str(row["heldout"]),
        str(row.get("line", "lodo_teacher")),
        int(row["seed"]),
    )
    config = dict(checkpoint["config"])
    config.update({
        "checkpoint_dir": "",
        "checkpoint_resume": False,
        "progress_logging": False,
    })
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(**config),
    )
    next_stage = algorithm._load_checkpoint_payload(checkpoint)
    if algorithm.task_ensemble is None:
        raise ValueError("terminal task audit requires a task ensemble")
    return algorithm, checkpoint, row, int(next_stage)


def _normalised_nearest_distance(problem, point, observed):
    if not observed:
        return None
    target = np.asarray(problem.normalize(point), dtype=float)
    rows = np.vstack([
        np.asarray(problem.normalize(x), dtype=float) for x in observed
    ])
    return float(np.min(np.linalg.norm(rows - target[None, :], axis=1)))


def _basis_nearest_distance(state, point, observed):
    if not observed:
        return None
    model = state.gpr_models[1]
    basis = getattr(model, "basis_map", None)
    if basis is None:
        return None
    rows = np.asarray(basis.features_many(observed), dtype=float)
    target = np.asarray(basis.features(point), dtype=float).reshape(1, -1)
    scale = np.std(rows, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return float(np.min(np.linalg.norm(
        (rows - target) / scale[None, :], axis=1)))


def _point_row(
    label,
    index,
    pool,
    algorithm,
    expert_names,
    con_mu,
    con_epistemic,
    con_aleatoric,
    expert_margin_mean,
    expert_violation,
    components,
    decision_margin,
    observed,
):
    x = pool[int(index)]
    observations = algorithm.observations.get(tuple(x), [])
    observed_constraints = [
        float(np.asarray(value, dtype=float)[1]) for value in observations
    ]
    return {
        "label": str(label),
        "index": int(index),
        "x": list(map(int, x)),
        "observed": bool(observed_constraints),
        "replicate_count": int(len(observed_constraints)),
        "observed_constraint_values": observed_constraints,
        "observed_constraint_mean": (
            None
            if not observed_constraints
            else float(np.mean(observed_constraints))
        ),
        "raw_nearest_observed_distance": _normalised_nearest_distance(
            algorithm.problem, x, observed),
        "basis_nearest_observed_distance": {
            state.name: _basis_nearest_distance(state, x, observed)
            for state in algorithm.task_ensemble.states
        },
        "decision_margin": float(decision_margin[index]),
        "bayes_objective": float(components["objective"][index]),
        "bayes_expected_violation": float(
            components["expected_violation"][index]),
        "bayes_risk": float(components["risk"][index]),
        "expert_predictions": {
            name: {
                "constraint_mean": float(con_mu[expert, index]),
                "epistemic_variance": float(
                    con_epistemic[expert, index]),
                "aleatoric_variance": float(
                    con_aleatoric[expert, index]),
                "chance_margin_mean": float(
                    expert_margin_mean[expert, index]),
                "expected_positive_violation": float(
                    expert_violation[expert, index]),
            }
            for expert, name in enumerate(expert_names)
        },
    }


def _rank_scores(values):
    values = np.asarray(values, dtype=float)
    ranks = np.empty_like(values, dtype=float)
    for row in range(values.shape[0]):
        order = np.argsort(values[row], kind="stable")
        ranks[row, order] = np.arange(values.shape[1], dtype=float)
    denominator = max(values.shape[1] - 1, 1)
    return np.mean(ranks / float(denominator), axis=0)


def run_audit(source_result, checkpoint_path):
    algorithm, checkpoint, source_row, next_stage = _restore(
        source_result, checkpoint_path)
    pool = [
        tuple(int(value) for value in x)
        for x in checkpoint.get("last_terminal_pool", [])
    ]
    selected = tuple(int(value) for value in source_row["x_recommended"])
    best_true = source_row.get("recommendation_best_true_feasible_x")
    best_true = (
        None
        if best_true is None
        else tuple(int(value) for value in best_true)
    )
    pool = list(dict.fromkeys(
        pool + [selected] + ([] if best_true is None else [best_true])
    ))
    if not pool:
        raise ValueError("final checkpoint has no terminal candidates")

    ensemble = algorithm.task_ensemble
    expert_names = list(ensemble.posterior.expert_names)
    con_mu, con_epistemic, con_aleatoric = ensemble.expert_moments_many(
        1, pool, certification=True)
    z_alpha = float(norm.ppf(1.0 - algorithm.problem.alpha))
    expert_margin_mean = (
        con_mu
        + z_alpha * np.sqrt(np.maximum(con_aleatoric, 0.0))
        - float(algorithm.problem.tau)
    )
    expert_violation = algorithm._normal_positive_part(
        expert_margin_mean,
        np.maximum(con_epistemic, 0.0),
    )
    components = algorithm._terminal_bayes_risk_components(
        algorithm.gpr,
        algorithm.variance_model,
        pool,
        task_ensemble=ensemble,
    )
    _, recommendation = algorithm._solve_posterior_recommendation(pool=pool)
    robust = ensemble.robust_moments_many(1, pool, certification=True)
    cert = algorithm._certification_result(
        robust.mean_upper,
        pool,
        robust.aleatoric_upper,
        epistemic=robust.epistemic_upper,
    )
    decision_margin = np.asarray(
        cert.margin + algorithm._recommendation_slack(), dtype=float)
    frozen_selected = int(np.argmin(components["risk"]))
    selected_index = pool.index(selected)
    best_index = None if best_true is None else pool.index(best_true)
    observed = [tuple(int(value) for value in x) for x in algorithm.observations]

    objective = np.asarray(components["objective"], dtype=float)
    robust_violation = np.asarray(
        components["expected_violation"], dtype=float)
    posterior_weights = np.asarray(
        ensemble.posterior.decision_weights(), dtype=float)
    prior_weights = np.asarray(
        ensemble.posterior.prior_weights(), dtype=float)
    posterior_violation = posterior_weights @ expert_violation
    prior_violation = prior_weights @ expert_violation
    median_violation = np.median(expert_violation, axis=0)
    sorted_violation = np.sort(expert_violation, axis=0)
    trimmed_violation = np.mean(sorted_violation[1:-1], axis=0)
    rank_consensus = _rank_scores(expert_violation)
    policy_indices = {
        "bayes_risk_penalty_5": int(np.argmin(components["risk"])),
        "minimum_robust_expected_violation": int(np.argmin(
            robust_violation)),
        "minimum_theory_margin": int(np.argmin(decision_margin)),
        "minimum_posterior_expected_violation": int(np.argmin(
            posterior_violation)),
        "minimum_prior_expected_violation": int(np.argmin(
            prior_violation)),
        "minimum_median_expert_violation": int(np.argmin(
            median_violation)),
        "minimum_trimmed_expert_violation": int(np.argmin(
            trimmed_violation)),
        "minimum_expert_rank_consensus": int(np.argmin(rank_consensus)),
    }
    for penalty in (10.0, 20.0, 50.0, 100.0):
        policy_indices[f"bayes_risk_penalty_{int(penalty)}"] = int(np.argmin(
            objective + penalty * robust_violation))

    # All posterior quantities above are frozen before this truth-only audit.
    true_margin = np.asarray([
        algorithm._true_chance_margin(x) for x in pool
    ], dtype=float)
    true_objective = np.asarray([
        algorithm.problem.true_objective(x) for x in pool
    ], dtype=float)
    true_feasible = true_margin <= 0.0
    weights = np.asarray(
        ensemble.posterior.posterior_weights(), dtype=float)
    decision_weights = np.asarray(
        ensemble.posterior.decision_weights(), dtype=float)
    expert_rows = []
    expert_safety_nominations = []
    expert_safety_topk = []
    safety_topk = min(16, len(pool))
    for expert, name in enumerate(expert_names):
        order = np.argsort(
            expert_violation[expert], kind="stable")
        nomination = int(order[0])
        feasible_ranks = np.flatnonzero(true_feasible[order])
        first_feasible_rank = (
            None if len(feasible_ranks) == 0
            else int(feasible_ranks[0]) + 1
        )
        expert_rows.append({
            "name": str(name),
            "posterior_weight": float(weights[expert]),
            "decision_weight": float(decision_weights[expert]),
            "true_feasibility_auc": _auc(
                true_feasible, -expert_violation[expert]),
            "mean_expected_positive_violation": float(np.mean(
                expert_violation[expert])),
            "first_true_feasible_rank": first_feasible_rank,
            "true_feasible_in_top_2": bool(np.any(
                true_feasible[order[:2]])),
            "true_feasible_in_top_4": bool(np.any(
                true_feasible[order[:4]])),
            "true_feasible_in_top_8": bool(np.any(
                true_feasible[order[:8]])),
            "true_feasible_in_top_16": bool(np.any(
                true_feasible[order[:16]])),
        })
        expert_safety_nominations.append({
            "name": str(name),
            "index": int(nomination),
            "x": list(map(int, pool[nomination])),
            "observed": bool(tuple(pool[nomination]) in algorithm.observations),
            "replicate_count": int(len(algorithm.observations.get(
                tuple(pool[nomination]), []))),
            "decision_weight": float(decision_weights[expert]),
            "expert_expected_positive_violation": float(
                expert_violation[expert, nomination]),
            "true_feasible": bool(true_feasible[nomination]),
            "true_chance_margin": float(true_margin[nomination]),
            "true_objective": float(true_objective[nomination]),
            "truth_joined_after_freeze": True,
            "admissible_decision_input": False,
        })
        expert_safety_topk.extend({
            "name": str(name),
            "rank": int(rank) + 1,
            "index": int(index),
            "x": list(map(int, pool[index])),
            "observed": bool(
                tuple(pool[index]) in algorithm.observations),
            "expert_expected_positive_violation": float(
                expert_violation[expert, index]),
            "true_feasible": bool(true_feasible[index]),
            "true_chance_margin": float(true_margin[index]),
            "true_objective": float(true_objective[index]),
            "truth_joined_after_freeze": True,
            "admissible_decision_input": False,
        } for rank, index in enumerate(order[:safety_topk]))
    policy_rows = []
    for name, index in policy_indices.items():
        policy_rows.append({
            "policy": str(name),
            "index": int(index),
            "x": list(map(int, pool[index])),
            "observed": bool(tuple(pool[index]) in algorithm.observations),
            "true_feasible": bool(true_feasible[index]),
            "true_chance_margin": float(true_margin[index]),
            "true_objective": float(true_objective[index]),
            "bayes_objective": float(objective[index]),
            "robust_expected_violation": float(robust_violation[index]),
            "posterior_expected_violation": float(
                posterior_violation[index]),
            "prior_expected_violation": float(prior_violation[index]),
            "median_expert_violation": float(median_violation[index]),
            "trimmed_expert_violation": float(trimmed_violation[index]),
            "expert_rank_consensus": float(rank_consensus[index]),
            "decision_margin": float(decision_margin[index]),
        })
    payload = {
        "schema_version": 1,
        "audit": "final_terminal_ranking",
        "status": "completed",
        "offline_only": True,
        "target_simulator_calls": 0,
        "oracle_used_for_selection": False,
        "admissible_decision_input": False,
        "heldout": str(source_row["heldout"]),
        "seed": int(source_row["seed"]),
        "checkpoint_reason": checkpoint.get("reason"),
        "checkpoint_next_stage_n": int(next_stage),
        "terminal_pool_size": int(len(pool)),
        "n_observed_policies": int(len(observed)),
        "n_observed_evaluations": int(sum(
            len(values) for values in algorithm.observations.values())),
        "selected_matches_frozen_bayes_risk": bool(
            selected_index == frozen_selected),
        "frozen_bayes_risk_x": list(map(int, pool[frozen_selected])),
        "ensemble_true_feasibility_auc": _auc(
            true_feasible, -components["expected_violation"]),
        "decision_margin_true_feasibility_auc": _auc(
            true_feasible, -decision_margin),
        "true_feasible_count": int(np.sum(true_feasible)),
        "expert_summary": expert_rows,
        "expert_safety_nominations": expert_safety_nominations,
        "expert_safety_topk": expert_safety_topk,
        "expert_safety_topk_limit": int(safety_topk),
        "policy_counterfactuals_after_freeze": policy_rows,
        "points": [
            _point_row(
                "selected",
                selected_index,
                pool,
                algorithm,
                expert_names,
                con_mu,
                con_epistemic,
                con_aleatoric,
                expert_margin_mean,
                expert_violation,
                components,
                decision_margin,
                observed,
            ),
            *(
                []
                if best_index is None
                else [_point_row(
                    "best_true_feasible",
                    best_index,
                    pool,
                    algorithm,
                    expert_names,
                    con_mu,
                    con_epistemic,
                    con_aleatoric,
                    expert_margin_mean,
                    expert_violation,
                    components,
                    decision_margin,
                    observed,
                )]
            ),
        ],
        "truth_after_freeze": {
            "selected_true_margin": float(true_margin[selected_index]),
            "selected_true_objective": float(true_objective[selected_index]),
            "selected_true_feasible": bool(true_feasible[selected_index]),
            "best_true_margin": (
                None if best_index is None else float(true_margin[best_index])
            ),
            "best_true_objective": (
                None
                if best_index is None
                else float(true_objective[best_index])
            ),
        },
        "recommendation_reproduced_x": recommendation.get(
            "x_recommended", list(map(int, pool[frozen_selected]))),
    }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = run_audit(args.source_result, args.checkpoint)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
