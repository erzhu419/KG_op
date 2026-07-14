"""Audit noise-limited certifiability and transferable-coordinate sufficiency.

This benchmark is intentionally diagnostic.  Oracle target margins may be
used by the representation-ceiling regressors, so none of its candidates are
eligible for promotion into the optimizer.  The strict source-frozen V4 prior
is reported separately as a reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.oracle_certification import (  # noqa: E402
    oracle_certifiability_metrics,
)
from performance import benchmark_tcb_v2_source_gate as shared  # noqa: E402
from performance import benchmark_tcb_v4_synthesis_gate as v4  # noqa: E402
from performance.benchmark_quality import json_safe, parse_csv  # noqa: E402
from representation.meta_prior import LearnedMetaPrior  # noqa: E402
from representation.observable_coordinate import (  # noqa: E402
    observable_policy_library,
)


def _truth(problem, points):
    z_alpha = float(norm.ppf(1.0 - problem.alpha))
    margin = np.asarray([
        float(problem.true_outputs(point)[1])
        + z_alpha * float(problem.true_sigma(point)[1])
        - float(problem.tau)
        for point in points
    ], dtype=float)
    sigma = np.asarray([
        float(problem.true_sigma(point)[1]) for point in points
    ], dtype=float)
    return margin, sigma


def _deduplicate(points, *, excluded=()):
    excluded = {tuple(int(v) for v in point) for point in excluded}
    rows = []
    seen = set(excluded)
    for point in points:
        key = tuple(int(v) for v in point)
        if key not in seen:
            seen.add(key)
            rows.append(key)
    return rows


def _cap_rows(rows, count, rng):
    rows = list(rows)
    if len(rows) <= int(count):
        return rows
    indices = rng.choice(len(rows), size=int(count), replace=False)
    return [rows[int(index)] for index in indices]


def _declared_hook_points(problem, count, rng, *, excluded=()):
    """Collect domain-declared candidates for an explicit oracle pool audit."""

    sources = {}
    calls = (
        ("initial_samples", lambda: problem.initial_samples(
            n=int(count), rng=rng)),
        ("structured_candidates", lambda: problem.structured_candidates(
            n=int(count), rng=rng)),
        ("recommendation_refinement", lambda: (
            problem.recommendation_refinement_candidates())),
        ("axis_solutions", lambda: problem.all_axis_solutions()),
    )
    combined = []
    for name, call in calls:
        try:
            rows = _cap_rows(call() or [], count, rng)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            rows = []
        rows = _deduplicate(rows, excluded=excluded)
        sources[name] = int(len(rows))
        combined.extend(rows)
    return _deduplicate(combined, excluded=excluded), sources


def _raw_descriptors(problem, points):
    return np.vstack([
        LearnedMetaPrior.descriptor(problem, point) for point in points
    ])


def _provider_rows(problem, points, *, coordinate):
    getter = (
        LearnedMetaPrior.provider_risk_coordinate
        if coordinate else LearnedMetaPrior.provider_risk_descriptor
    )
    rows = [getter(problem, point) for point in points]
    if not rows or any(row is None for row in rows):
        return None
    return np.vstack(rows)


def _projector(args, mode, source_raw):
    options = dict(vars(args))
    options["descriptor_mode"] = str(mode)
    return shared._BoundaryDescriptorProjector(
        SimpleNamespace(**options)).fit(source_raw)


def _coordinate_bank(
    args,
    problem,
    candidate_points,
    evaluation_points,
    source_rows,
    source_model,
):
    source_raw = np.vstack([row["descriptor"] for row in source_rows])
    candidate_raw = _raw_descriptors(problem, candidate_points)
    evaluation_raw = _raw_descriptors(problem, evaluation_points)
    bank = {
        "raw": {
            "candidate": candidate_raw,
            "evaluation": evaluation_raw,
            "stratum": "source_frozen_observable",
        },
        "observable_multiscale": {
            "candidate": np.vstack([
                observable_policy_library(problem, point)
                for point in candidate_points
            ]),
            "evaluation": np.vstack([
                observable_policy_library(problem, point)
                for point in evaluation_points
            ]),
            "stratum": "source_frozen_observable",
        },
    }

    for mode in ("learned_coordinate", "learned_risk"):
        projector = _projector(args, mode, source_raw)
        candidate = projector.transform(candidate_raw)
        evaluation = projector.transform(evaluation_raw)
        bank[mode] = {
            "candidate": candidate,
            "evaluation": evaluation,
            "stratum": "source_frozen_observable",
        }
        bank[f"raw+{mode}"] = {
            "candidate": np.column_stack([candidate_raw, candidate]),
            "evaluation": np.column_stack([evaluation_raw, evaluation]),
            "stratum": "source_frozen_observable",
        }

    candidate_provider_descriptor = [
        LearnedMetaPrior.provider_risk_descriptor(problem, point)
        for point in candidate_points
    ]
    candidate_provider_coordinate = [
        LearnedMetaPrior.provider_risk_coordinate(problem, point)
        for point in candidate_points
    ]
    evaluation_provider_descriptor = [
        LearnedMetaPrior.provider_risk_descriptor(problem, point)
        for point in evaluation_points
    ]
    evaluation_provider_coordinate = [
        LearnedMetaPrior.provider_risk_coordinate(problem, point)
        for point in evaluation_points
    ]
    source_candidate = source_model.boundary_descriptor_projector_.transform(
        candidate_raw,
        candidate_provider_descriptor,
        candidate_provider_coordinate,
    )
    source_evaluation = source_model.boundary_descriptor_projector_.transform(
        evaluation_raw,
        evaluation_provider_descriptor,
        evaluation_provider_coordinate,
    )
    candidate_atoms = source_model._family_scores(source_candidate)
    evaluation_atoms = source_model._family_scores(source_evaluation)
    bank["source_atom_scores"] = {
        "candidate": candidate_atoms,
        "evaluation": evaluation_atoms,
        "stratum": "source_frozen_observable",
    }
    bank["raw+source_atom_scores"] = {
        "candidate": np.column_stack([candidate_raw, candidate_atoms]),
        "evaluation": np.column_stack([evaluation_raw, evaluation_atoms]),
        "stratum": "source_frozen_observable",
    }

    candidate_provider = _provider_rows(
        problem, candidate_points, coordinate=True)
    evaluation_provider = _provider_rows(
        problem, evaluation_points, coordinate=True)
    if candidate_provider is not None and evaluation_provider is not None:
        bank["provider_coordinate"] = {
            "candidate": candidate_provider,
            "evaluation": evaluation_provider,
            "stratum": "domain_tuned_oracle_upper_bound",
        }
        bank["raw+provider_coordinate"] = {
            "candidate": np.column_stack([
                candidate_raw, candidate_provider]),
            "evaluation": np.column_stack([
                evaluation_raw, evaluation_provider]),
            "stratum": "domain_tuned_oracle_upper_bound",
        }
    candidate_risk = _provider_rows(
        problem, candidate_points, coordinate=False)
    evaluation_risk = _provider_rows(
        problem, evaluation_points, coordinate=False)
    if candidate_risk is not None and evaluation_risk is not None:
        bank["provider_risk"] = {
            "candidate": candidate_risk,
            "evaluation": evaluation_risk,
            "stratum": "domain_tuned_oracle_upper_bound",
        }
        bank["raw+provider_risk"] = {
            "candidate": np.column_stack([candidate_raw, candidate_risk]),
            "evaluation": np.column_stack([evaluation_raw, evaluation_risk]),
            "stratum": "domain_tuned_oracle_upper_bound",
        }
    return bank, source_candidate, source_evaluation


def _selection_indices(margins, count, policy, rng):
    margins = np.asarray(margins, dtype=float)
    count = min(max(1, int(count)), len(margins))
    if policy == "random":
        return np.asarray(rng.choice(
            len(margins), size=count, replace=False), dtype=int)
    if policy != "oracle_boundary_stratified":
        raise ValueError(f"unknown oracle training policy {policy!r}")
    selected = []
    seen = set()

    def add(indices, quota):
        for index in indices:
            index = int(index)
            if len(selected) >= count or quota <= 0:
                break
            if index not in seen:
                seen.add(index)
                selected.append(index)
                quota -= 1

    boundary_quota = int(np.ceil(0.5 * count))
    side_quota = int(np.ceil(0.25 * count))
    add(np.argsort(np.abs(margins), kind="stable"), boundary_quota)
    safe = np.where(margins <= 0.0)[0]
    unsafe = np.where(margins > 0.0)[0]
    add(safe[np.argsort(-margins[safe], kind="stable")], side_quota)
    add(unsafe[np.argsort(margins[unsafe], kind="stable")], side_quota)
    add(np.argsort(margins, kind="stable"), count)
    add(np.argsort(-margins, kind="stable"), count)
    return np.asarray(selected[:count], dtype=int)


def _split_indices(count, rng):
    if int(count) < 5:
        raise ValueError("oracle diagnostic needs at least five target rows")
    order = np.asarray(rng.permutation(int(count)), dtype=int)
    calibration_count = max(1, int(np.floor(0.2 * count)))
    validation_count = max(1, int(np.floor(0.2 * count)))
    fit_count = int(count) - calibration_count - validation_count
    if fit_count < 3:
        fit_count = 3
        validation_count = 1
        calibration_count = int(count) - fit_count - validation_count
    fit = order[:fit_count]
    validation = order[fit_count:fit_count + validation_count]
    calibration = order[fit_count + validation_count:]
    return fit, validation, calibration


def _scaler(x):
    x = np.asarray(x, dtype=float)
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return center, scale


def _fit_linear(x, y, ridge, *, nonnegative=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    center, scale = _scaler(x)
    z = (x - center) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.diag(np.concatenate([[0.0], np.full(z.shape[1], ridge)]))
    if nonnegative:
        augmented = np.vstack([design, np.sqrt(penalty)])
        target = np.concatenate([y, np.zeros(design.shape[1])])
        theta = lsq_linear(
            augmented,
            target,
            bounds=(
                np.concatenate([[-np.inf], np.zeros(z.shape[1])]),
                np.full(design.shape[1], np.inf),
            ),
        ).x
    else:
        theta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y

    def predict(new_x):
        new_z = (np.asarray(new_x, dtype=float) - center) / scale
        return np.column_stack([
            np.ones(len(new_z)), new_z]) @ theta

    return predict


def _fit_rbf(x, y, ridge, lengthscale):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    center, scale = _scaler(x)
    z = (x - center) / scale
    lengthscale = max(float(lengthscale), 1e-6)
    kernel = np.exp(-0.5 * cdist(z, z, metric="sqeuclidean")
                    / (lengthscale ** 2))
    target_center = float(np.mean(y))
    coefficients = np.linalg.pinv(
        kernel + max(float(ridge), 1e-10) * np.eye(len(z))) @ (
            y - target_center)

    def predict(new_x):
        new_z = (np.asarray(new_x, dtype=float) - center) / scale
        cross = np.exp(-0.5 * cdist(
            new_z, z, metric="sqeuclidean") / (lengthscale ** 2))
        return target_center + cross @ coefficients

    return predict


def _model_candidates(kind, x_fit):
    ridges = (1e-6, 1e-3, 1e-1)
    if kind in {"ridge_linear", "nonnegative_synthesis"}:
        return [{"ridge": ridge} for ridge in ridges]
    if kind != "rbf_kernel_ridge":
        raise ValueError(f"unknown diagnostic regressor {kind!r}")
    center, scale = _scaler(x_fit)
    z = (np.asarray(x_fit, dtype=float) - center) / scale
    distances = pdist(z)
    positive = distances[distances > 1e-10]
    median = float(np.median(positive)) if len(positive) else 1.0
    return [
        {"ridge": ridge, "lengthscale": median * multiplier}
        for multiplier in (0.5, 1.0, 2.0)
        for ridge in ridges
    ]


def _fit_kind(kind, x, y, options):
    if kind == "ridge_linear":
        return _fit_linear(x, y, options["ridge"])
    if kind == "nonnegative_synthesis":
        return _fit_linear(
            x, y, options["ridge"], nonnegative=True)
    return _fit_rbf(
        x, y, options["ridge"], options["lengthscale"])


def _target_oracle_fit(kind, x, y, evaluation_x, rng, upper_alpha):
    fit, validation, calibration = _split_indices(len(y), rng)
    best = None
    for options in _model_candidates(kind, x[fit]):
        model = _fit_kind(kind, x[fit], y[fit], options)
        prediction = model(x[validation])
        loss = float(np.mean((prediction - y[validation]) ** 2))
        key = (loss, tuple(sorted(options.items())))
        if best is None or key < best[0]:
            best = (key, options)
    refit = np.concatenate([fit, validation])
    model = _fit_kind(kind, x[refit], y[refit], best[1])
    mean = np.asarray(model(evaluation_x), dtype=float)
    calibration_residual = np.asarray(
        y[calibration] - model(x[calibration]), dtype=float)
    order = np.sort(calibration_residual)
    conformal_rank = int(np.ceil(
        (len(order) + 1) * (1.0 - float(upper_alpha))))
    finite_sample_supported = conformal_rank <= len(order)
    index = min(max(conformal_rank, 1), len(order)) - 1
    guard = max(float(order[index]), 0.0)
    return mean, mean + guard, {
        "selected_hyperparameters": best[1],
        "validation_mse": float(best[0][0]),
        "fit_count": int(len(fit)),
        "validation_count": int(len(validation)),
        "calibration_count": int(len(calibration)),
        "one_sided_calibration_guard": guard,
        "conformal_rank": conformal_rank,
        "finite_sample_conformal_supported": bool(
            finite_sample_supported),
        "target_oracle_used_for_fit": True,
        "target_oracle_used_for_hyperparameter_selection": True,
    }


def _aliasing_metrics(coordinate, true_margin, neighbors=5):
    coordinate = np.asarray(coordinate, dtype=float)
    margin = np.asarray(true_margin, dtype=float)
    center, scale = _scaler(coordinate)
    z = (coordinate - center) / scale
    distances = cdist(z, z, metric="sqeuclidean")
    np.fill_diagonal(distances, np.inf)
    count = min(max(1, int(neighbors)), max(len(z) - 1, 1))
    nearest = np.argpartition(distances, kth=count - 1, axis=1)[:, :count]
    discrepancy = np.mean(np.abs(
        margin[:, None] - margin[nearest]), axis=1)
    boundary_count = min(
        len(margin), max(8, int(np.ceil(0.25 * len(margin)))))
    boundary = np.argsort(np.abs(margin), kind="stable")[:boundary_count]
    margin_scale = max(float(np.std(margin)), 1e-8)
    return {
        "neighbor_count": int(count),
        "mean_neighbor_margin_discrepancy": float(np.mean(discrepancy)),
        "boundary_neighbor_margin_discrepancy": float(np.mean(
            discrepancy[boundary])),
        "normalized_neighbor_discrepancy": float(
            np.mean(discrepancy) / margin_scale),
        "normalized_boundary_neighbor_discrepancy": float(
            np.mean(discrepancy[boundary]) / margin_scale),
        "target_oracle_used_for_metric": True,
    }


def _extended_metrics(true_margin, mean, upper):
    metrics = shared._prediction_metrics(true_margin, mean, upper)
    margin = np.asarray(true_margin, dtype=float)
    mean = np.asarray(mean, dtype=float)
    boundary_count = min(
        len(margin), max(8, int(np.ceil(0.25 * len(margin)))))
    boundary = np.argsort(np.abs(margin), kind="stable")[:boundary_count]
    scale = max(float(np.std(margin)), 1e-8)
    metrics.update({
        "normalized_margin_rmse": float(metrics["margin_rmse"] / scale),
        "normalized_boundary_mae": float(metrics["boundary_mae"] / scale),
        "safe_sign_accuracy": float(np.mean(
            (mean <= 0.0) == (margin <= 0.0))),
        "boundary_safe_sign_accuracy": float(np.mean(
            (mean[boundary] <= 0.0) == (margin[boundary] <= 0.0))),
        "true_margin_standard_deviation": scale,
    })
    return metrics


def run_audit(args):
    heldout = str(args.heldout)
    problem = shared._problem(
        heldout, d=args.d, L=args.L, sigma=args.sigma, alpha=args.alpha)
    rng = np.random.default_rng(shared._stable_seed(
        args.target_seed, f"{heldout}:certifiability"))
    random_points = shared._unique_random_points(
        problem, args.oracle_candidate_pool + args.evaluation_pool, rng)
    candidate_points = list(random_points[:args.oracle_candidate_pool])
    evaluation_points = list(random_points[args.oracle_candidate_pool:])
    hooks, hook_counts = _declared_hook_points(
        problem,
        args.hook_pool_per_source,
        rng,
        excluded=random_points,
    )
    candidate_points = _deduplicate(candidate_points + hooks)
    candidate_margin, candidate_sigma = _truth(problem, candidate_points)
    evaluation_margin, evaluation_sigma = _truth(problem, evaluation_points)
    augmented_points = _deduplicate(candidate_points + evaluation_points)
    augmented_margin, augmented_sigma = _truth(problem, augmented_points)

    if str(args.descriptor_mode) != "raw+learned_risk":
        raise ValueError(
            "the audit preregisters raw+learned_risk source atoms")
    source_model, source_domains, rows_by_domain = shared.fit_source_model(
        args,
        heldout,
        model_builder=v4._fit_synthesis_model_from_rows,
    )
    source_rows = [rows_by_domain[domain] for domain in source_domains]
    bank, _, source_evaluation = _coordinate_bank(
        args,
        problem,
        candidate_points,
        evaluation_points,
        source_rows,
        source_model,
    )

    prior = source_model.prior_adapter()
    strict_mean = source_model.predict(source_evaluation, adapter=prior)
    strict_upper = source_model.predict_upper(source_evaluation, adapter=prior)
    rows = [{
        "heldout": heldout,
        "target_seed": int(args.target_seed),
        "coordinate": "source_atom_scores",
        "coordinate_dimension": int(len(source_model.families_)),
        "coordinate_stratum": "source_frozen_observable",
        "fit_stratum": "strict_source_frozen",
        "model_kind": "v4_source_prior",
        "training_policy": "source_only",
        "target_train_count": 0,
        "target_oracle_used_for_fit": False,
        "target_oracle_used_for_hyperparameter_selection": False,
        "evaluation_oracle_used_after_fit": True,
        "promotion_eligible": False,
        "metrics": _extended_metrics(
            evaluation_margin, strict_mean, strict_upper),
    }]

    train_sizes = tuple(sorted(set(int(value) for value in args.train_sizes)))
    policies = tuple(args.training_policies)
    regressors = tuple(args.regressors)
    aliasing = {}
    for coordinate_name, coordinate_rows in bank.items():
        candidate_coordinate = np.asarray(
            coordinate_rows["candidate"], dtype=float)
        evaluation_coordinate = np.asarray(
            coordinate_rows["evaluation"], dtype=float)
        aliasing[coordinate_name] = _aliasing_metrics(
            evaluation_coordinate,
            evaluation_margin,
            neighbors=args.aliasing_neighbors,
        )
        coordinate_regressors = list(regressors)
        if coordinate_name == "source_atom_scores":
            coordinate_regressors.append("nonnegative_synthesis")
        coordinate_regressors = list(dict.fromkeys(coordinate_regressors))
        for train_size in train_sizes:
            if train_size > len(candidate_points):
                continue
            for policy in policies:
                selection_rng = np.random.default_rng(shared._stable_seed(
                    args.target_seed,
                    f"{heldout}:{coordinate_name}:{policy}:{train_size}",
                ))
                selection_margin = (
                    candidate_margin[:args.oracle_candidate_pool]
                    if policy == "random" else candidate_margin
                )
                selected = _selection_indices(
                    selection_margin,
                    train_size,
                    policy,
                    selection_rng,
                )
                x_train = candidate_coordinate[selected]
                y_train = candidate_margin[selected]
                for kind in coordinate_regressors:
                    fit_rng = np.random.default_rng(shared._stable_seed(
                        args.target_seed,
                        f"{heldout}:{coordinate_name}:{policy}:"
                        f"{train_size}:{kind}",
                    ))
                    mean, upper, diagnostics = _target_oracle_fit(
                        kind,
                        x_train,
                        y_train,
                        evaluation_coordinate,
                        fit_rng,
                        args.upper_alpha,
                    )
                    rows.append({
                        "heldout": heldout,
                        "target_seed": int(args.target_seed),
                        "coordinate": coordinate_name,
                        "coordinate_dimension": int(
                            candidate_coordinate.shape[1]),
                        "coordinate_stratum": coordinate_rows["stratum"],
                        "fit_stratum": "target_oracle_diagnostic",
                        "model_kind": kind,
                        "training_policy": policy,
                        "training_candidate_pool": (
                            "uniform_random"
                            if policy == "random"
                            else "domain_augmented_oracle_pool"
                        ),
                        "target_train_count": int(train_size),
                        "target_oracle_used_for_fit": True,
                        "target_oracle_used_for_hyperparameter_selection": True,
                        "target_oracle_used_for_training_selection": bool(
                            policy == "oracle_boundary_stratified"),
                        "evaluation_oracle_used_after_fit": True,
                        "promotion_eligible": False,
                        "fit_diagnostics": diagnostics,
                        "metrics": _extended_metrics(
                            evaluation_margin, mean, upper),
                    })

    return {
        "schema_version": 1,
        "audit": "noise_certifiability_and_coordinate_sufficiency",
        "heldout": heldout,
        "target_seed": int(args.target_seed),
        "source_seed": int(args.source_seed),
        "source_domains": list(source_domains),
        "source_atom_descriptor_mode": str(args.descriptor_mode),
        "random_evaluation_pool_size": int(len(evaluation_points)),
        "oracle_candidate_pool_size": int(len(candidate_points)),
        "declared_hook_counts": hook_counts,
        "certifiability": {
            "uniform_random": oracle_certifiability_metrics(
                evaluation_margin,
                evaluation_sigma,
                args.replicate_budgets,
                confidence_alpha=args.upper_alpha,
            ),
            "domain_augmented_oracle_pool": oracle_certifiability_metrics(
                augmented_margin,
                augmented_sigma,
                args.replicate_budgets,
                confidence_alpha=args.upper_alpha,
            ),
        },
        "coordinate_aliasing": aliasing,
        "rows": rows,
        "source_model": source_model.diagnostics(),
        "leakage_contract": {
            "outer_target_excluded_from_source_model": True,
            "strict_reference_uses_target_outcomes": False,
            "oracle_regressors_use_target_truth": True,
            "provider_coordinates_are_domain_tuned_upper_bounds": True,
            "all_audit_rows_promotion_eligible": False,
            "uniform_evaluation_points_disjoint_from_target_training_pool": True,
            "offline_only": True,
        },
    }


def build_parser():
    parser = v4.build_parser()
    parser.description = (
        "Noise certifiability and source-coordinate sufficiency audit")
    parser.set_defaults(
        descriptor_mode="raw+learned_risk",
        geometry="low_rank_psd",
        rank=2,
        coefficient_ridge=0.1,
        coefficient_prior_strength=0.5,
    )
    parser.add_argument("--oracle-candidate-pool", type=int, default=512)
    parser.add_argument("--hook-pool-per-source", type=int, default=512)
    parser.add_argument("--train-sizes", default="10,20,40,80")
    parser.add_argument(
        "--training-policies", default="random,oracle_boundary_stratified")
    parser.add_argument("--regressors", default="ridge_linear,rbf_kernel_ridge")
    parser.add_argument("--replicate-budgets", default="1,3,5,10,20,50,100")
    parser.add_argument("--aliasing-neighbors", type=int, default=5)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.domains = tuple(parse_csv(args.domains))
    args.pilot_policies = tuple(parse_csv(args.pilot_policies))
    args.training_policies = tuple(parse_csv(args.training_policies))
    args.regressors = tuple(parse_csv(args.regressors))
    args.train_sizes = tuple(int(value) for value in parse_csv(args.train_sizes))
    args.replicate_budgets = tuple(
        int(value) for value in parse_csv(args.replicate_budgets))
    if args.heldout not in args.domains:
        raise ValueError("heldout domain must be present in --domains")
    result = run_audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(
        f"DONE certifiability_coordinate_audit rows={len(result['rows'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
