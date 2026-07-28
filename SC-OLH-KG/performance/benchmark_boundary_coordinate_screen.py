"""Nested source-only screen for transferable chance-boundary posteriors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau, norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_quality import json_safe  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import LearnedMetaPrior  # noqa: E402
from representation.transferable_boundary import (  # noqa: E402
    TransferableChanceBoundaryPosterior,
)


SOURCE_BANK = (
    "RZDT1",
    "RZDT2",
    "RegimeRZDT1",
    "StatePolicyRZDT1",
    "HighDimStatePolicyRZDT1",
)


def _progress(stage, **payload):
    print(json.dumps({
        "event": "tcb_screen_progress",
        "stage": str(stage),
        **payload,
    }, sort_keys=True), flush=True)


@dataclass(frozen=True)
class BoundaryRows:
    descriptors: np.ndarray
    margins: np.ndarray


@dataclass(frozen=True)
class DomainBoundaryDataset:
    training: BoundaryRows
    pilot: BoundaryRows
    pilot_pool: BoundaryRows
    evaluation: BoundaryRows


def _stable_seed(label, seed):
    digest = hashlib.sha256(f"{label}:{int(seed)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def _problem(name, *, sigma, alpha):
    d = 1000 if name == "HighDimStatePolicyRZDT1" else 12
    return ScalarizedProblem(
        make_problem(name, d=d, L=100, sigma=sigma, alpha=alpha),
        weights=(0.5, 0.5),
    )


def _random_rows(problem, count, rng):
    descriptors = []
    margins = []
    z_alpha = float(norm.ppf(1.0 - problem.alpha))
    for _ in range(max(1, int(count))):
        x = tuple(int(v) for v in problem.sample_random(rng))
        sigma = float(problem.true_sigma(x)[1])
        raw_margin = (
            float(problem.true_constraint_mean(x))
            + z_alpha * sigma
            - float(problem.tau)
        )
        scale = max(float(problem.sigma_level), sigma, 1e-6)
        descriptors.append(LearnedMetaPrior.descriptor(problem, x))
        margins.append(raw_margin / scale)
    return BoundaryRows(
        np.asarray(descriptors, dtype=float),
        np.asarray(margins, dtype=float),
    )


def _boundary_stratified(rows, count):
    margins = rows.margins
    count = min(max(1, int(count)), len(margins))
    selected = []
    seen = set()

    def add(order, limit):
        for index in order:
            index = int(index)
            if len(selected) >= count or limit <= 0:
                break
            if index in seen:
                continue
            seen.add(index)
            selected.append(index)
            limit -= 1

    boundary_count = int(np.ceil(0.5 * count))
    side_count = int(np.ceil(0.2 * count))
    add(np.argsort(np.abs(margins), kind="stable"), boundary_count)
    safe = np.where(margins <= 0.0)[0]
    unsafe = np.where(margins > 0.0)[0]
    add(safe[np.argsort(-margins[safe], kind="stable")], side_count)
    add(unsafe[np.argsort(margins[unsafe], kind="stable")], side_count)
    add(np.argsort(margins, kind="stable"), count)
    add(np.argsort(-margins, kind="stable"), count)
    index = np.asarray(selected[:count], dtype=int)
    return BoundaryRows(rows.descriptors[index], rows.margins[index])


def make_domain_dataset(
    name,
    *,
    records_per_domain,
    pilot_size,
    evaluation_size,
    pool_multiplier,
    sigma,
    alpha,
    seed,
):
    problem = _problem(name, sigma=sigma, alpha=alpha)
    training_pool = _random_rows(
        problem,
        records_per_domain * pool_multiplier,
        np.random.default_rng(_stable_seed(f"{name}:train", seed)),
    )
    pilot = _random_rows(
        problem,
        pilot_size,
        np.random.default_rng(_stable_seed(f"{name}:pilot", seed)),
    )
    pilot_pool = _random_rows(
        problem,
        max(pilot_size, pilot_size * pool_multiplier),
        np.random.default_rng(_stable_seed(f"{name}:pilot_pool", seed)),
    )
    evaluation_pool = _random_rows(
        problem,
        evaluation_size * pool_multiplier,
        np.random.default_rng(_stable_seed(f"{name}:eval", seed)),
    )
    return DomainBoundaryDataset(
        training=_boundary_stratified(training_pool, records_per_domain),
        pilot=pilot,
        pilot_pool=pilot_pool,
        evaluation=_boundary_stratified(evaluation_pool, evaluation_size),
    )


def _rank_loss(truth, prediction):
    tau = kendalltau(
        np.asarray(truth, dtype=float),
        np.asarray(prediction, dtype=float),
        nan_policy="omit",
    ).statistic
    if tau is None or not np.isfinite(tau):
        return 1.0
    return float(0.5 * (1.0 - tau))


def _source_boundary_pilot_indices(model, descriptors, count, *, robust=False):
    descriptors = np.asarray(descriptors, dtype=float)
    count = min(max(1, int(count)), len(descriptors))
    coordinate = model._coordinate(descriptors)
    score = np.asarray(model.predict(descriptors), dtype=float)
    if robust:
        score_quantiles = np.quantile(score, [0.05, 0.10, 0.50, 0.90, 0.95])
        score_scale = max(
            float(score_quantiles[3] - score_quantiles[1]) / 2.563,
            1e-8,
        )
        standardized_score = np.clip(
            (score - float(score_quantiles[2])) / score_scale,
            -3.0,
            3.0,
        )
        eligible = (
            (score >= float(score_quantiles[0]))
            & (score <= float(score_quantiles[4]))
        )
        coordinate_center = np.median(coordinate, axis=0)
        coordinate_low, coordinate_high = np.quantile(
            coordinate, [0.10, 0.90], axis=0)
        coordinate_scale = np.maximum(
            (coordinate_high - coordinate_low) / 2.563,
            1e-8,
        )
        design_coordinate = np.clip(
            (coordinate - coordinate_center) / coordinate_scale,
            -3.0,
            3.0,
        )
    else:
        score_quantiles = np.quantile(score, [0.0, 0.0, 0.5, 1.0, 1.0])
        score_scale = max(float(np.std(score)), 1e-8)
        standardized_score = (score - float(np.mean(score))) / score_scale
        eligible = np.ones(len(score), dtype=bool)
        design_coordinate = coordinate
    feature_rank = min(2, coordinate.shape[1])
    design = np.column_stack([
        np.ones(len(score), dtype=float),
        standardized_score,
        design_coordinate[:, :feature_rank],
    ])
    design_scale = np.sqrt(np.mean(design ** 2, axis=0))
    design = design / np.where(design_scale < 1e-8, 1.0, design_scale)
    eligible_indices = np.where(eligible)[0]
    seed_targets = (
        [score_quantiles[1], 0.0, score_quantiles[3]]
        if robust
        else [float(np.min(score)), 0.0, float(np.max(score))]
    )
    seed_indices = [
        int(eligible_indices[np.argmin(np.abs(
            score[eligible_indices] - float(target)
        ))])
        for target in seed_targets
    ]
    selected = []
    for index in seed_indices:
        if index not in selected and len(selected) < count:
            selected.append(index)
    information = 1e-3 * np.eye(design.shape[1], dtype=float)
    for index in selected:
        information += np.outer(design[index], design[index])
    while len(selected) < count:
        inverse = np.linalg.pinv(information)
        leverage = np.einsum("ni,ij,nj->n", design, inverse, design)
        boundary_bonus = np.exp(-0.5 * standardized_score ** 2)
        utility = np.log1p(np.maximum(leverage, 0.0)) + 0.5 * boundary_bonus
        utility[~eligible] = -np.inf
        utility[np.asarray(selected, dtype=int)] = -np.inf
        if not np.any(np.isfinite(utility)):
            utility = np.log1p(np.maximum(leverage, 0.0))
            utility[np.asarray(selected, dtype=int)] = -np.inf
        index = int(np.argmax(utility))
        selected.append(index)
        information += np.outer(design[index], design[index])
    selected = np.asarray(selected, dtype=int)
    return selected, {
        "mode": (
            "source_boundary_robust_d_optimal"
            if robust else "source_boundary_d_optimal"),
        "selected_count": int(len(selected)),
        "source_prediction_min": float(np.min(score[selected])),
        "source_prediction_max": float(np.max(score[selected])),
        "source_prediction_nearest_boundary": float(np.min(
            np.abs(score[selected]))),
        "design_rank": int(np.linalg.matrix_rank(design[selected])),
        "source_support_lower": float(score_quantiles[0]),
        "source_support_upper": float(score_quantiles[4]),
        "robust_support_clipping": bool(robust),
        "target_outcomes_used_for_selection": False,
        "target_oracle_used": False,
    }


def fold_metrics(model, target, heldout, *, pilot=None, pilot_selection=None):
    pilot = target.pilot if pilot is None else pilot
    pilot_selection = pilot_selection or {
        "mode": "unstratified_random_before_truth",
        "selected_count": int(len(pilot.margins)),
        "target_outcomes_used_for_selection": False,
        "target_oracle_used": False,
    }
    adapter = model.fit_target_adapter(
        pilot.descriptors,
        pilot.margins,
    )
    prediction = model.predict(
        target.evaluation.descriptors, adapter=adapter)
    upper = model.predict_upper(
        target.evaluation.descriptors, adapter=adapter)
    truth = target.evaluation.margins
    boundary_weight = 1.0 + 2.0 * np.exp(-0.5 * truth ** 2)
    false_safe = (upper <= 0.0) & (truth > 0.0)
    false_unsafe = (upper > 0.0) & (truth <= 0.0)
    nominal_false_safe = (prediction <= 0.0) & (truth > 0.0)
    predicted_safe = upper <= 0.0
    true_safe = truth <= 0.0
    true_unsafe = ~true_safe
    safe_count = int(np.sum(true_safe))
    unsafe_count = int(np.sum(true_unsafe))
    predicted_safe_count = int(np.sum(predicted_safe))
    return {
        "heldout": str(heldout),
        "n_pilot": int(len(pilot.margins)),
        "n_evaluation": int(len(truth)),
        "pilot_feasible_rate": float(np.mean(pilot.margins <= 0.0)),
        "evaluation_feasible_rate": float(np.mean(truth <= 0.0)),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "boundary_rmse": float(np.sqrt(np.average(
            (prediction - truth) ** 2,
            weights=boundary_weight,
        ))),
        "rank_loss": _rank_loss(truth, prediction),
        "false_safe_rate": float(np.mean(false_safe)),
        "nominal_false_safe_rate": float(np.mean(nominal_false_safe)),
        "false_unsafe_rate": float(np.mean(false_unsafe)),
        "certified_safe_rate": float(np.mean(predicted_safe)),
        "safe_recall": float(
            np.sum(predicted_safe & true_safe) / safe_count
            if safe_count else 1.0),
        "dangerous_recall": float(
            np.sum((~predicted_safe) & true_unsafe) / unsafe_count
            if unsafe_count else 1.0),
        "certified_safe_precision": float(
            np.sum(predicted_safe & true_safe) / predicted_safe_count
            if predicted_safe_count else 1.0),
        "upper_coverage": float(np.mean(upper >= truth)),
        "prediction_finite": bool(
            np.all(np.isfinite(prediction)) and np.all(np.isfinite(upper))),
        "adapter": adapter.diagnostics,
        "pilot_selection": pilot_selection,
        "target_evaluation_used_for_fit": False,
        "target_oracle_used": False,
    }


def _dataset_job(payload):
    name, options = payload
    return name, make_domain_dataset(name, **options)


def _fold_job(payload):
    heldout, datasets, model_options, protocol_options = payload
    descriptors = []
    margins = []
    domains = []
    for source in SOURCE_BANK:
        if source == heldout:
            continue
        rows = datasets[source].training
        descriptors.append(rows.descriptors)
        margins.append(rows.margins)
        domains.extend([source] * len(rows.margins))
    model = TransferableChanceBoundaryPosterior(**model_options).fit(
        np.vstack(descriptors),
        np.concatenate(margins),
        np.asarray(domains, dtype=object),
    )
    target = datasets[heldout]
    if protocol_options["pilot_selection_mode"] in {
        "source_boundary_d_optimal",
        "source_boundary_robust_d_optimal",
    }:
        indices, pilot_selection = _source_boundary_pilot_indices(
            model,
            target.pilot_pool.descriptors,
            protocol_options["pilot_size"],
            robust=(
                protocol_options["pilot_selection_mode"]
                == "source_boundary_robust_d_optimal"
            ),
        )
        pilot = BoundaryRows(
            target.pilot_pool.descriptors[indices],
            target.pilot_pool.margins[indices],
        )
    else:
        pilot = target.pilot
        pilot_selection = None
    row = fold_metrics(
        model,
        target,
        heldout,
        pilot=pilot,
        pilot_selection=pilot_selection,
    )
    row["source_fit"] = model.diagnostics()
    return heldout, row


def _parallel_map(jobs, fn, payloads):
    jobs = min(max(1, int(jobs)), len(payloads))
    if jobs == 1:
        for payload in payloads:
            yield fn(payload)
        return
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
    ) as executor:
        futures = [executor.submit(fn, payload) for payload in payloads]
        for future in as_completed(futures):
            yield future.result()


def run_screen(args):
    dataset_options = {
        "records_per_domain": args.records_per_domain,
        "pilot_size": args.pilot_size,
        "evaluation_size": args.evaluation_size,
        "pool_multiplier": args.pool_multiplier,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "seed": args.data_seed,
    }
    _progress(
        "dataset_batch_start",
        total=len(SOURCE_BANK),
        jobs=min(max(1, int(args.jobs)), len(SOURCE_BANK)),
    )
    datasets = {}
    for complete, (name, dataset) in enumerate(_parallel_map(
        args.jobs,
        _dataset_job,
        [(name, dataset_options) for name in SOURCE_BANK],
    ), start=1):
        datasets[name] = dataset
        _progress(
            "dataset_complete",
            current=complete,
            total=len(SOURCE_BANK),
            domain=name,
        )
    model_options = {
        "coordinate": args.coordinate,
        "geometry": args.geometry,
        "adaptation": args.adaptation,
        "rank": args.rank,
        "ridge": args.ridge,
        "domain_penalty": args.domain_penalty,
        "boundary_temperature": args.boundary_temperature,
        "adaptation_ridge": args.adaptation_ridge,
        "upper_alpha": args.upper_alpha,
        "calibration_prior_df": args.calibration_prior_df,
        "target_residual_rank": args.target_residual_rank,
    }
    _progress(
        "fold_batch_start",
        total=len(SOURCE_BANK),
        jobs=min(max(1, int(args.jobs)), len(SOURCE_BANK)),
    )
    folds_by_domain = {}
    fold_payloads = [
        (
            heldout,
            datasets,
            model_options,
            {
                "pilot_selection_mode": args.pilot_selection_mode,
                "pilot_size": args.pilot_size,
            },
        )
        for heldout in SOURCE_BANK
    ]
    for complete, (heldout, row) in enumerate(_parallel_map(
        args.jobs,
        _fold_job,
        fold_payloads,
    ), start=1):
        folds_by_domain[heldout] = row
        _progress(
            "fold_complete",
            current=complete,
            total=len(SOURCE_BANK),
            heldout=heldout,
            false_safe_rate=row["false_safe_rate"],
            boundary_rmse=row["boundary_rmse"],
        )
    folds = [folds_by_domain[heldout] for heldout in SOURCE_BANK]
    false_safe = np.asarray([row["false_safe_rate"] for row in folds])
    rank_loss = np.asarray([row["rank_loss"] for row in folds])
    boundary_rmse = np.asarray([row["boundary_rmse"] for row in folds])
    coverage = np.asarray([row["upper_coverage"] for row in folds])
    false_unsafe = np.asarray([row["false_unsafe_rate"] for row in folds])
    certified_safe = np.asarray([
        row["certified_safe_rate"] for row in folds])
    safe_recall = np.asarray([row["safe_recall"] for row in folds])
    dangerous_recall = np.asarray([
        row["dangerous_recall"] for row in folds])
    adaptation_dims = np.asarray([
        row["adapter"].get("effective_label_adaptation_dimension", 0)
        for row in folds
    ], dtype=int)
    adaptation_cap = 0.35 * float(args.pilot_size)
    return {
        "schema_version": 1,
        "variant": {
            "coordinate": args.coordinate,
            "geometry": args.geometry,
            "adaptation": args.adaptation,
            "rank": int(args.rank),
        },
        "protocol": {
            "kind": "nested_source_bank_lodo",
            "source_bank": list(SOURCE_BANK),
            "online_sentinel_domains_used": False,
            "records_per_domain": int(args.records_per_domain),
            "pilot_size": int(args.pilot_size),
            "evaluation_size": int(args.evaluation_size),
            "data_seed": int(args.data_seed),
            "parallel_jobs": int(args.jobs),
            "predictive_upper_alpha": float(args.upper_alpha),
            "calibration_prior_df": float(args.calibration_prior_df),
            "target_residual_rank": int(args.target_residual_rank),
            "target_pilot_selection": str(args.pilot_selection_mode),
            "target_pilot_selection_outcomes_used": False,
            "target_evaluation_selection": "fixed_boundary_stress_set",
            "target_evaluation_used_for_fit": False,
            "target_oracle_used": False,
        },
        "folds": folds,
        "aggregate": {
            "all_finite": bool(all(row["prediction_finite"] for row in folds)),
            "worst_false_safe_rate": float(np.max(false_safe)),
            "mean_false_safe_rate": float(np.mean(false_safe)),
            "median_rank_loss": float(np.median(rank_loss)),
            "median_boundary_rmse": float(np.median(boundary_rmse)),
            "minimum_upper_coverage": float(np.min(coverage)),
            "mean_upper_coverage": float(np.mean(coverage)),
            "mean_false_unsafe_rate": float(np.mean(false_unsafe)),
            "mean_certified_safe_rate": float(np.mean(certified_safe)),
            "mean_safe_recall": float(np.mean(safe_recall)),
            "minimum_dangerous_recall": float(np.min(dangerous_recall)),
            "nonvacuous_safe_fold_count": int(sum(
                row["evaluation_feasible_rate"] > 0.05
                and row["certified_safe_rate"] > 0.0
                for row in folds
            )),
            "effective_adaptation_dimension_max": int(np.max(adaptation_dims)),
            "adaptation_dimension_cap": float(adaptation_cap),
            "adaptation_dimension_admissible": bool(
                np.max(adaptation_dims) <= adaptation_cap + 1e-12),
            "single_domain_collapse": bool(
                np.max(false_safe) > 0.25
                or np.max(boundary_rmse) > 3.0 * max(
                    float(np.median(boundary_rmse)), 1e-12)
            ),
        },
    }


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coordinate",
        choices=TransferableChanceBoundaryPosterior.COORDINATES,
        required=True,
    )
    parser.add_argument(
        "--geometry",
        choices=TransferableChanceBoundaryPosterior.GEOMETRIES,
        required=True,
    )
    parser.add_argument(
        "--adaptation",
        choices=TransferableChanceBoundaryPosterior.ADAPTATIONS,
        required=True,
    )
    parser.add_argument("--rank", type=int, choices=(2, 4), required=True)
    parser.add_argument("--records-per-domain", type=int, default=96)
    parser.add_argument("--pilot-size", type=int, default=10)
    parser.add_argument("--evaluation-size", type=int, default=48)
    parser.add_argument("--pool-multiplier", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--domain-penalty", type=float, default=0.5)
    parser.add_argument("--boundary-temperature", type=float, default=1.0)
    parser.add_argument("--adaptation-ridge", type=float, default=5.0)
    parser.add_argument("--upper-alpha", type=float, default=0.01)
    parser.add_argument("--calibration-prior-df", type=float, default=1.0)
    parser.add_argument("--target-residual-rank", type=int, default=1)
    parser.add_argument(
        "--pilot-selection-mode",
        choices=(
            "random",
            "source_boundary_d_optimal",
            "source_boundary_robust_d_optimal",
        ),
        default="random",
    )
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--data-seed", type=int, default=33001)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _progress(
        "start",
        coordinate=args.coordinate,
        geometry=args.geometry,
        adaptation=args.adaptation,
        rank=int(args.rank),
    )
    payload = run_screen(args)
    atomic_write(args.out, payload)
    print(json.dumps({
        "schema_version": int(payload["schema_version"]),
        "status": "complete",
        "variant": payload["variant"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
