"""Oracle-only identifiability audit for ordered cumulative coordinates.

Candidate generation and all feature maps are frozen before target truth is
read.  The resulting target-oracle regression is an upper-bound diagnostic,
never an optimization result or an admissible decision input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.candidates import unique_candidates  # noqa: E402
from core.cumulative_risk import cumulative_feature_vector  # noqa: E402
from performance.benchmark_lodo_meta_prior import build_target_problem  # noqa: E402
from performance.benchmark_quality import json_safe  # noqa: E402
from performance.run_lodo_manifest_shard import load_config  # noqa: E402


def _ridge_predict(train_x, train_y, test_x, ridge):
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    X = (train_x - mean) / scale
    T = (test_x - mean) / scale
    X = np.column_stack([np.ones(len(X)), X])
    T = np.column_stack([np.ones(len(T)), T])
    penalty = float(ridge) * np.eye(X.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(X.T @ X + penalty) @ X.T @ train_y
    return T @ beta


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _auc(feasible, safety_score):
    feasible = np.asarray(feasible, dtype=bool)
    n_pos = int(np.sum(feasible))
    n_neg = int(len(feasible) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(safety_score)
    return float((
        np.sum(ranks[feasible]) - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg))


def _fit_audit(features, margin, train, validation, test):
    ridges = (1e-4, 1e-2, 1.0, 10.0, 100.0)
    selected = None
    for ridge in ridges:
        pred = _ridge_predict(
            features[train], margin[train], features[validation], ridge)
        mse = float(np.mean((pred - margin[validation]) ** 2))
        dangerous = float(np.mean(np.maximum(
            margin[validation] - pred, 0.0) ** 2))
        score = mse + 2.0 * dangerous
        candidate = (score, ridge)
        if selected is None or candidate < selected:
            selected = candidate
    ridge = float(selected[1])
    fit_indices = np.concatenate([train, validation])
    prediction = _ridge_predict(
        features[fit_indices], margin[fit_indices], features[test], ridge)
    truth = margin[test]
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 1.0 - float(np.sum((prediction - truth) ** 2)) / max(
        denominator, 1e-12)
    feasible = truth <= 0.0
    return {
        "feature_dim": int(features.shape[1]),
        "feature_rank": int(np.linalg.matrix_rank(features[fit_indices])),
        "selected_ridge": ridge,
        "test_r2": float(r2),
        "test_feasible_count": int(np.sum(feasible)),
        "test_auc": _auc(feasible, -prediction),
        "predicted_feasible_count": int(np.sum(prediction <= 0.0)),
        "false_feasible_rate": float(np.mean(
            (prediction <= 0.0) & (truth > 0.0))),
    }


def _r2_score(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return 1.0 - float(np.sum((prediction - truth) ** 2)) / max(
        denominator, 1e-12)


def _fit_decomposed_audit(
    mean_features,
    variance_features,
    mean_margin,
    variance,
    z_value,
    train,
    validation,
    test,
):
    ridges = (1e-4, 1e-2, 1.0, 10.0, 100.0)
    selected = None
    for mean_ridge in ridges:
        mean_prediction = _ridge_predict(
            mean_features[train], mean_margin[train],
            mean_features[validation], mean_ridge)
        for variance_ridge in ridges:
            variance_prediction = _ridge_predict(
                variance_features[train], variance[train],
                variance_features[validation], variance_ridge)
            chance_prediction = mean_prediction + float(z_value) * np.sqrt(
                np.maximum(variance_prediction, 0.0))
            chance_truth = mean_margin[validation] + float(z_value) * np.sqrt(
                np.maximum(variance[validation], 0.0))
            mse = float(np.mean((chance_prediction - chance_truth) ** 2))
            dangerous = float(np.mean(np.maximum(
                chance_truth - chance_prediction, 0.0) ** 2))
            candidate = (
                mse + 2.0 * dangerous,
                float(mean_ridge),
                float(variance_ridge),
            )
            if selected is None or candidate < selected:
                selected = candidate
    fit_indices = np.concatenate([train, validation])
    mean_prediction = _ridge_predict(
        mean_features[fit_indices], mean_margin[fit_indices],
        mean_features[test], selected[1])
    variance_prediction = _ridge_predict(
        variance_features[fit_indices], variance[fit_indices],
        variance_features[test], selected[2])
    variance_prediction = np.maximum(variance_prediction, 0.0)
    chance_prediction = mean_prediction + float(z_value) * np.sqrt(
        variance_prediction)
    chance_truth = mean_margin[test] + float(z_value) * np.sqrt(
        np.maximum(variance[test], 0.0))
    feasible = chance_truth <= 0.0
    return {
        "mean_feature_dim": int(mean_features.shape[1]),
        "variance_feature_dim": int(variance_features.shape[1]),
        "mean_feature_rank": int(np.linalg.matrix_rank(
            mean_features[fit_indices])),
        "variance_feature_rank": int(np.linalg.matrix_rank(
            variance_features[fit_indices])),
        "selected_mean_ridge": float(selected[1]),
        "selected_variance_ridge": float(selected[2]),
        "test_mean_r2": float(_r2_score(
            mean_margin[test], mean_prediction)),
        "test_variance_r2": float(_r2_score(
            variance[test], variance_prediction)),
        "test_chance_r2": float(_r2_score(
            chance_truth, chance_prediction)),
        "test_auc": _auc(feasible, -chance_prediction),
        "predicted_feasible_count": int(np.sum(chance_prediction <= 0.0)),
        "false_feasible_rate": float(np.mean(
            (chance_prediction <= 0.0) & (chance_truth > 0.0))),
    }


def _ordered_group_feature_maps(exposures):
    A = np.vstack([np.asarray(item.A, dtype=float) for item in exposures])
    N = np.vstack([np.asarray(item.N, dtype=float) for item in exposures])
    a_energy = np.sum(A ** 2, axis=1, keepdims=True)
    n_sum = np.sum(N, axis=1, keepdims=True)
    n_energy = np.sum(N ** 2, axis=1, keepdims=True)
    return {
        "ordered_fully_invariant": np.column_stack([
            np.linalg.norm(A, axis=1),
            a_energy[:, 0],
            np.linalg.norm(N, axis=1),
            n_energy[:, 0],
        ]),
        "ordered_curvature_grouped": np.column_stack([
            A,
            a_energy,
            N,
        ]),
        "ordered_shared_grouped": np.column_stack([
            A,
            A ** 2,
            n_sum,
            n_energy,
        ]),
        "ordered_both_grouped": np.column_stack([
            A,
            a_energy,
            n_sum,
            n_energy,
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--heldout", default="InventorySupplyChain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=1024)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = load_config(args.manifest)
    config.update({
        "meta_ordered_cumulative_exposure": True,
        "meta_ordered_exposure_max_frequency": 8,
        "meta_ordered_exposure_active_dim": 2,
        "meta_ordered_exposure_frequency_penalty": 0.10,
        "offline_only": True,
    })
    problem, _ = build_target_problem(
        config, args.heldout, "lodo_teacher", args.seed)
    prior = problem.meta_prior
    rng = np.random.default_rng(20260711 + int(args.seed))
    candidates = [
        problem.sample_random(rng) for _ in range(max(1, args.pool_size))
    ]
    candidates.extend(prior.universal_shape_candidates(
        problem, n=128, rng=rng, force=True))
    candidates.extend(prior.profile_template_candidates(
        problem, n=128, rng=rng))
    candidates.extend(prior.proposal_candidates(
        problem, n=128, rng=rng, pool_size=max(512, args.pool_size)))
    candidates = unique_candidates(candidates)

    ordered_exposures = [
        prior.ordered_cumulative_risk_exposure(problem, x)
        for x in candidates
    ]
    ordered_linear = np.vstack([
        np.concatenate([exposure.A, exposure.N])
        for exposure in ordered_exposures
    ])
    ordered_diagonal = np.vstack([
        np.concatenate([exposure.A, exposure.A ** 2, exposure.N])
        for exposure in ordered_exposures
    ])
    feature_maps = {
        "aggregate_coordinate": np.vstack([
            prior.coordinate_basis_features(problem, x) for x in candidates
        ]),
        "aligned_coordinate": np.vstack([
            prior.frozen_risk_aligned_coordinate(problem, x)
            for x in candidates
        ]),
        "ordered_linear": ordered_linear,
        "ordered_diagonal_quadratic": ordered_diagonal,
        "ordered_cumulative": np.vstack([
            prior.ordered_coordinate_basis_features(problem, x)
            for x in candidates
        ]),
        "full_descriptor": np.vstack([
            prior._scaled_descriptor(prior.descriptor(problem, x))
            for x in candidates
        ]),
        **_ordered_group_feature_maps(ordered_exposures),
    }

    cumulative_variance = np.vstack([
        cumulative_feature_vector(exposure)
        for exposure in ordered_exposures
    ])
    grouped = _ordered_group_feature_maps(ordered_exposures)
    n_local = len(ordered_exposures[0].A)
    decomposed_feature_pairs = {
        "linear_mean__cumulative_factor_variance": (
            ordered_linear,
            cumulative_variance,
        ),
        "linear_mean__shared_grouped_variance": (
            ordered_linear,
            np.column_stack([
                np.ones(len(ordered_linear)),
                ordered_linear[:, :n_local] ** 2,
                grouped["ordered_both_grouped"][:, -2:],
            ]),
        ),
        "diagonal_mean__cumulative_factor_variance": (
            ordered_diagonal,
            cumulative_variance,
        ),
    }

    # Oracle truth is joined only after candidates and representations freeze.
    mean_margin = np.asarray([
        problem.base.true_constraint_mean(x) - float(problem.tau)
        for x in candidates
    ], dtype=float)
    variance = np.asarray([
        float(problem.base.true_sigma(x)[1]) ** 2
        for x in candidates
    ], dtype=float)
    z_value = float(norm.ppf(1.0 - problem.alpha))
    margin = mean_margin + z_value * np.sqrt(np.maximum(variance, 0.0))
    order = rng.permutation(len(candidates))
    n_train = max(3, int(0.60 * len(order)))
    n_validation = max(2, int(0.20 * len(order)))
    train = order[:n_train]
    validation = order[n_train:n_train + n_validation]
    test = order[n_train + n_validation:]
    payload = {
        "schema_version": 1,
        "oracle_only": True,
        "admissible_decision_input": False,
        "target_simulator_calls": 0,
        "heldout": str(args.heldout),
        "seed": int(args.seed),
        "n_candidates": int(len(candidates)),
        "true_feasible_rate": float(np.mean(margin <= 0.0)),
        "ordered_diagnostics": dict(prior.ordered_exposure_diagnostics),
        "representations": {
            name: _fit_audit(values, margin, train, validation, test)
            for name, values in feature_maps.items()
        },
        "decomposed_representations": {
            name: _fit_decomposed_audit(
                mean_features,
                variance_features,
                mean_margin,
                variance,
                z_value,
                train,
                validation,
                test,
            )
            for name, (mean_features, variance_features)
            in decomposed_feature_pairs.items()
        },
    }
    encoded = json.dumps(json_safe(payload), indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
