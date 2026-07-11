"""Offline group-complexity audit from a trusted final runtime checkpoint.

The audit reuses only the target observations already charged to one KG run.
Candidates, representations, and LOO-selected penalties are frozen before
analytic synthetic truth is joined.  Oracle-selected rows are reported only as
upper-bound diagnostics and are never admissible decision inputs.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.candidates import unique_candidates  # noqa: E402
from performance.benchmark_lodo_meta_prior import build_target_problem  # noqa: E402
from performance.benchmark_quality import json_safe  # noqa: E402
from performance.diagnose_ordered_coordinate import _auc, _r2_score  # noqa: E402
from performance.run_lodo_manifest_shard import load_config  # noqa: E402


RIDGES = (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _fit_penalized_predict(train_x, train_y, test_x, penalty):
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float).reshape(-1)
    penalty = np.asarray(penalty, dtype=float).reshape(-1)
    if train_x.ndim != 2 or train_x.shape[1] != len(penalty):
        raise ValueError("penalty must provide one value per feature")
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    X = (train_x - mean) / scale
    T = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(X)), X])
    test_design = np.column_stack([np.ones(len(T)), T])
    precision = np.diag(np.concatenate([[0.0], penalty]))
    system = design.T @ design + precision
    beta = np.linalg.pinv(system) @ design.T @ train_y
    return test_design @ beta


def _effective_degrees_of_freedom(features, penalty):
    features = np.asarray(features, dtype=float)
    penalty = np.asarray(penalty, dtype=float).reshape(-1)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    Z = (features - np.mean(features, axis=0)) / scale
    gram = Z.T @ Z
    smoother = gram @ np.linalg.pinv(gram + np.diag(penalty))
    return float(np.trace(smoother))


def _loo_loss(features, response, penalty):
    features = np.asarray(features, dtype=float)
    response = np.asarray(response, dtype=float).reshape(-1)
    predictions = []
    for heldout in range(len(response)):
        train = np.arange(len(response)) != heldout
        prediction = _fit_penalized_predict(
            features[train], response[train],
            features[heldout:heldout + 1], penalty,
        )
        predictions.append(float(prediction[0]))
    predictions = np.asarray(predictions, dtype=float)
    residual = response - predictions
    mse = float(np.mean(residual ** 2))
    dangerous = float(np.mean(np.maximum(residual, 0.0) ** 2))
    return mse + 2.0 * dangerous


def _group_penalties(group_sizes, ridges=RIDGES):
    rows = []
    for values in product(tuple(ridges), repeat=len(group_sizes)):
        penalty = np.concatenate([
            np.full(int(size), float(value), dtype=float)
            for size, value in zip(group_sizes, values)
        ])
        rows.append((tuple(map(float, values)), penalty))
    return rows


def _evaluate_prediction(predicted_mean, true_mean, true_sigma, tau, z_value):
    predicted_margin = (
        np.asarray(predicted_mean, dtype=float)
        - float(tau)
        + float(z_value) * np.asarray(true_sigma, dtype=float)
    )
    true_margin = (
        np.asarray(true_mean, dtype=float)
        - float(tau)
        + float(z_value) * np.asarray(true_sigma, dtype=float)
    )
    feasible = true_margin <= 0.0
    dangerous = np.maximum(true_margin - predicted_margin, 0.0)
    return {
        "chance_rmse": float(np.sqrt(np.mean(
            (predicted_margin - true_margin) ** 2))),
        "dangerous_chance_rmse": float(np.sqrt(np.mean(dangerous ** 2))),
        "chance_r2": float(_r2_score(true_margin, predicted_margin)),
        "chance_auc": _auc(feasible, -predicted_margin),
        "predicted_feasible_count": int(np.sum(predicted_margin <= 0.0)),
        "false_feasible_rate": float(np.mean(
            (predicted_margin <= 0.0) & ~feasible)),
        "true_feasible_count": int(np.sum(feasible)),
    }


def _fit_family(
    train_features,
    response,
    test_features,
    penalties,
    true_mean,
    true_sigma,
    tau,
    z_value,
):
    scored = []
    for label, penalty in penalties:
        loo = _loo_loss(train_features, response, penalty)
        prediction = _fit_penalized_predict(
            train_features, response, test_features, penalty)
        metrics = _evaluate_prediction(
            prediction, true_mean, true_sigma, tau, z_value)
        effective_df = _effective_degrees_of_freedom(
            train_features, penalty)
        oracle_loss = (
            metrics["chance_rmse"]
            + 2.0 * metrics["dangerous_chance_rmse"]
        )
        scored.append({
            "penalty_label": list(label),
            "penalty": np.asarray(penalty, dtype=float).tolist(),
            "loo_loss": float(loo),
            "effective_df": float(effective_df),
            "metrics": metrics,
            "oracle_loss": float(oracle_loss),
        })
    selected = min(scored, key=lambda row: (
        row["loo_loss"], row["effective_df"], row["penalty_label"]))
    oracle = min(scored, key=lambda row: (
        row["oracle_loss"], row["effective_df"], row["penalty_label"]))
    return {
        "feature_dim": int(np.asarray(train_features).shape[1]),
        "train_feature_rank": int(np.linalg.matrix_rank(train_features)),
        "n_penalty_models": int(len(scored)),
        "selected_by_nested_loo": selected,
        "oracle_best_after_freeze": oracle,
        "oracle_used_for_selection": False,
    }


def _observed_response(observations):
    xs = []
    ys = []
    for x, values in observations.items():
        rows = np.asarray(values, dtype=float)
        xs.append(tuple(int(value) for value in x))
        ys.append(float(np.mean(rows, axis=0)[1]))
    return xs, np.asarray(ys, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with Path(args.checkpoint).open("rb") as handle:
        checkpoint = pickle.load(handle)
    observations = checkpoint.get("observations", {})
    observed_x, observed_y = _observed_response(observations)
    if len(observed_x) < 6:
        raise ValueError("complexity audit needs at least six observed policies")

    config = load_config(args.manifest)
    config.update({
        "meta_ordered_cumulative_exposure": True,
        "meta_ordered_exposure_max_frequency": 8,
        "meta_ordered_exposure_active_dim": 2,
        "meta_ordered_exposure_frequency_penalty": 0.10,
        "meta_ordered_exposure_basis_mode": "diagonal_quadratic",
        "meta_ordered_exposure_adaptive_sparsity": True,
        "meta_ordered_exposure_replace_local_kernel": False,
        "meta_ordered_exposure_semiparametric_residual": False,
        "meta_ordered_exposure_latent_structure_selection": True,
        "meta_ordered_exposure_group_shared_shrinkage": True,
        "offline_only": True,
    })
    problem, _ = build_target_problem(
        config, args.heldout, "lodo_teacher", args.seed)
    prior = problem.meta_prior
    rng = np.random.default_rng(20260711 + int(args.seed))

    candidates = list(checkpoint.get("last_terminal_pool", []))
    candidates.extend(observed_x)
    candidates.extend(
        problem.sample_random(rng) for _ in range(max(1, args.pool_size)))
    candidates.extend(prior.universal_shape_candidates(
        problem, n=128, rng=rng, force=True))
    candidates.extend(prior.profile_template_candidates(
        problem, n=128, rng=rng))
    candidates.extend(prior.proposal_candidates(
        problem, n=128, rng=rng,
        pool_size=max(512, int(args.pool_size))))
    candidates = unique_candidates(candidates)
    observed_set = set(observed_x)
    test_candidates = [x for x in candidates if tuple(x) not in observed_set]
    if len(test_candidates) < 20:
        raise ValueError("complexity audit candidate pool is too small")

    observed_exposures = [
        prior.ordered_cumulative_risk_exposure(problem, x)
        for x in observed_x
    ]
    test_exposures = [
        prior.ordered_cumulative_risk_exposure(problem, x)
        for x in test_candidates
    ]
    observed_A = np.vstack([item.A for item in observed_exposures])
    observed_N = np.vstack([item.N for item in observed_exposures])
    test_A = np.vstack([item.A for item in test_exposures])
    test_N = np.vstack([item.N for item in test_exposures])

    feature_families = {
        "linear_shared_scalar_ridge": (
            np.column_stack([observed_A, observed_N]),
            np.column_stack([test_A, test_N]),
            _group_penalties([observed_A.shape[1] + observed_N.shape[1]], RIDGES),
        ),
        "full_diagonal_scalar_ridge": (
            np.column_stack([observed_A, observed_A ** 2, observed_N]),
            np.column_stack([test_A, test_A ** 2, test_N]),
            _group_penalties([
                2 * observed_A.shape[1] + observed_N.shape[1]], RIDGES),
        ),
        "full_diagonal_group_ridge": (
            np.column_stack([observed_A, observed_A ** 2, observed_N]),
            np.column_stack([test_A, test_A ** 2, test_N]),
            _group_penalties([
                observed_A.shape[1],
                observed_A.shape[1],
                observed_N.shape[1],
            ], RIDGES),
        ),
        "fixed2_diagonal_group_ridge": (
            np.column_stack([
                observed_A[:, :2], observed_A ** 2, observed_N]),
            np.column_stack([test_A[:, :2], test_A ** 2, test_N]),
            _group_penalties([
                min(2, observed_A.shape[1]),
                observed_A.shape[1],
                observed_N.shape[1],
            ], RIDGES),
        ),
    }

    # Truth is joined only after candidate and model families are frozen.
    true_mean = np.asarray([
        problem.base.true_constraint_mean(x) for x in test_candidates
    ], dtype=float)
    true_sigma = np.asarray([
        float(problem.base.true_sigma(x)[1]) for x in test_candidates
    ], dtype=float)
    z_value = float(norm.ppf(1.0 - problem.alpha))
    observed_true_margin = np.asarray([
        problem.base.true_constraint_mean(x)
        - float(problem.tau)
        + z_value * float(problem.base.true_sigma(x)[1])
        for x in observed_x
    ], dtype=float)

    payload = {
        "schema_version": 1,
        "audit": "checkpoint_group_complexity",
        "oracle_only": True,
        "admissible_decision_input": False,
        "target_simulator_calls": 0,
        "heldout": str(args.heldout),
        "seed": int(args.seed),
        "checkpoint_reason": checkpoint.get("reason"),
        "checkpoint_next_stage_n": checkpoint.get("next_stage_n"),
        "n_observed_policies": int(len(observed_x)),
        "n_observed_evaluations": int(sum(
            len(values) for values in observations.values())),
        "n_test_candidates": int(len(test_candidates)),
        "observed_true_feasible_count": int(np.sum(
            observed_true_margin <= 0.0)),
        "observed_true_infeasible_count": int(np.sum(
            observed_true_margin > 0.0)),
        "families": {
            name: _fit_family(
                train_features,
                observed_y,
                test_features,
                penalties,
                true_mean,
                true_sigma,
                problem.tau,
                z_value,
            )
            for name, (train_features, test_features, penalties)
            in feature_families.items()
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
