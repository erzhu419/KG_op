"""Source-only LODO gate for the hierarchical TCB-V2 certificate.

Target truth is used only after fitting to score coverage and ranking.  The
target adapter sees ordinary noisy replicates from a source-proposed or random
pilot design and has exactly two free label-space parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zlib

import numpy as np
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_quality import json_safe, parse_csv  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import LearnedMetaPrior  # noqa: E402
from core.cumulative_risk import canonical_risk_descriptor  # noqa: E402
from representation.transferable_boundary import (  # noqa: E402
    HierarchicalSignedDistancePosterior,
)


DEFAULT_DOMAINS = (
    "RZDT1",
    "StatePolicyRZDT1",
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


class _BoundaryDescriptorProjector:
    """Source-frozen map from observable policy summaries to TCB inputs."""

    def __init__(self, args):
        self.mode = str(args.descriptor_mode).lower()
        if self.mode not in LearnedMetaPrior.VALID_BOUNDARY_DESCRIPTOR_MODES:
            raise ValueError(f"unknown boundary descriptor mode {self.mode!r}")
        self.meta_prior = None
        if "learned_" in self.mode:
            self.meta_prior = LearnedMetaPrior(
                local_dim=int(args.meta_local_dim),
                shared_dim=int(args.meta_shared_dim),
                soft_temperature=float(args.meta_soft_temperature),
                seed=int(args.source_seed),
            )

    def fit(self, raw_descriptors):
        raw = np.asarray(raw_descriptors, dtype=float)
        if self.meta_prior is not None:
            coordinate = self.meta_prior._fit_scaler_pca(raw)
            self.meta_prior._fit_kmeans(coordinate)
            self.meta_prior.fit_status = "boundary_descriptor_fit"
        return self

    def transform(
        self,
        raw_descriptors,
        provider_descriptors=None,
        provider_coordinates=None,
    ):
        raw = np.asarray(raw_descriptors, dtype=float)
        if raw.ndim == 1:
            raw = raw[None, :]
        provider_descriptors = (
            [None] * len(raw)
            if provider_descriptors is None
            else list(provider_descriptors)
        )
        if len(provider_descriptors) != len(raw):
            raise ValueError("provider descriptors must match raw rows")
        provider_coordinates = (
            [None] * len(raw)
            if provider_coordinates is None
            else list(provider_coordinates)
        )
        if len(provider_coordinates) != len(raw):
            raise ValueError("provider coordinates must match raw rows")
        rows = []
        for descriptor, provider, provider_coordinate in zip(
            raw, provider_descriptors, provider_coordinates,
        ):
            if "learned_" in self.mode:
                exposure = self.meta_prior.risk_exposure_from_descriptor(
                    descriptor)
                risk = (
                    np.concatenate([exposure.A, exposure.N]).astype(float)
                    if "learned_coordinate" in self.mode
                    else canonical_risk_descriptor(exposure)
                )
            elif "provider_" in self.mode:
                provider_value = (
                    provider_coordinate
                    if "provider_coordinate" in self.mode else provider
                )
                if provider_value is None:
                    raise ValueError(
                        "provider gate includes a domain without a declared "
                        "CumulativeRiskFeatureProvider")
                risk = np.asarray(provider_value, dtype=float).reshape(-1)
            else:
                risk = None
            if self.mode == "raw":
                rows.append(descriptor)
            elif not self.mode.startswith("raw+"):
                rows.append(risk)
            else:
                rows.append(np.concatenate([descriptor, risk]))
        return np.vstack(rows)


def _stable_seed(seed, label):
    return int(np.random.SeedSequence([
        int(seed) & 0xFFFFFFFF,
        zlib.crc32(str(label).encode("utf-8")) & 0xFFFFFFFF,
    ]).generate_state(1)[0])


def _problem(name, *, d, L, sigma, alpha):
    return ScalarizedProblem(
        make_problem(name, d=d, L=L, sigma=sigma, alpha=alpha))


def _unique_random_points(problem, count, rng):
    points = []
    seen = set()
    attempts = 0
    limit = max(100, 20 * int(count))
    while len(points) < int(count) and attempts < limit:
        point = tuple(int(v) for v in problem.sample_random(rng))
        attempts += 1
        if point not in seen:
            seen.add(point)
            points.append(point)
    if len(points) < int(count):
        raise RuntimeError(
            f"could sample only {len(points)} unique points of {count}")
    return points


def _replicate_margin(problem, point, rng, replicates, prior_df=2.0):
    values = np.vstack([
        np.asarray(problem.simulate(point, rng), dtype=float)
        for _ in range(max(1, int(replicates)))
    ])
    constraint = values[:, 1]
    prior_variance = max(float(problem.sigma_level) ** 2, 1e-8)
    if len(constraint) >= 2:
        residual_sum = float(np.sum(
            (constraint - float(np.mean(constraint))) ** 2))
        variance = (
            residual_sum + max(float(prior_df), 0.0) * prior_variance
        ) / max(len(constraint) - 1 + max(float(prior_df), 0.0), 1.0)
    else:
        variance = prior_variance
    variance = max(float(variance), 1e-10)
    margin = (
        float(np.mean(constraint))
        + float(norm.ppf(1.0 - problem.alpha)) * np.sqrt(variance)
        - float(problem.tau)
    )
    return margin, variance


def _source_rows(
    domain,
    *,
    count,
    replicates,
    source_seed,
    d,
    L,
    sigma,
    alpha,
):
    problem = _problem(domain, d=d, L=L, sigma=sigma, alpha=alpha)
    rng = np.random.default_rng(_stable_seed(source_seed, domain))
    points = _unique_random_points(problem, count, rng)
    descriptor = []
    provider_descriptor = []
    provider_coordinate = []
    margin = []
    variance = []
    for point in points:
        observed_margin, observed_variance = _replicate_margin(
            problem, point, rng, replicates)
        descriptor.append(LearnedMetaPrior.descriptor(problem, point))
        provider_descriptor.append(
            LearnedMetaPrior.provider_risk_descriptor(problem, point))
        provider_coordinate.append(
            LearnedMetaPrior.provider_risk_coordinate(problem, point))
        margin.append(observed_margin)
        variance.append(observed_variance)
    return {
        "descriptor": np.vstack(descriptor),
        "provider_risk_descriptor": provider_descriptor,
        "provider_risk_coordinate": provider_coordinate,
        "margin": np.asarray(margin, dtype=float),
        "variance": np.asarray(variance, dtype=float),
        "replicate_count": np.full(len(points), replicates, dtype=float),
        "domain": np.full(len(points), str(domain), dtype=object),
    }


def _fit_model_from_rows(args, rows):
    raw_descriptors = np.vstack([row["descriptor"] for row in rows])
    provider_descriptors = [
        descriptor
        for row in rows
        for descriptor in row["provider_risk_descriptor"]
    ]
    provider_coordinates = [
        coordinate
        for row in rows
        for coordinate in row["provider_risk_coordinate"]
    ]
    projector = _BoundaryDescriptorProjector(args).fit(raw_descriptors)
    descriptors = projector.transform(
        raw_descriptors, provider_descriptors, provider_coordinates)
    margins = np.concatenate([row["margin"] for row in rows])
    variances = np.concatenate([row["variance"] for row in rows])
    replicate_count = np.concatenate([
        row["replicate_count"] for row in rows])
    domains = np.concatenate([row["domain"] for row in rows])
    model = HierarchicalSignedDistancePosterior(
        coordinate=args.coordinate,
        geometry=args.geometry,
        rank=args.rank,
        ridge=args.ridge,
        domain_penalty=args.domain_penalty,
        boundary_temperature=args.boundary_temperature,
        adaptation_ridge=args.adaptation_ridge,
        upper_alpha=args.upper_alpha,
        calibration_prior_df=args.calibration_prior_df,
        hierarchy_iterations=args.hierarchy_iterations,
        effect_ridge=args.effect_ridge,
        rotation_mode=args.rotation_mode,
        rotation_ridge=args.rotation_ridge,
        target_residual_rank=args.target_residual_rank,
        residual_ridge=args.residual_ridge,
    )
    model.fit(
        descriptors,
        margins,
        domains,
        margin_variance=variances,
        replicate_count=replicate_count,
    )
    model.boundary_descriptor_projector_ = projector
    model.diagnostics_.update({
        "descriptor_mode": projector.mode,
        "descriptor_dimension": int(descriptors.shape[1]),
        "provider_structural_input": bool(
            "provider_risk" in projector.mode),
    })
    return model


def _project_row(model, row, row_slice=slice(None)):
    raw = np.asarray(row["descriptor"])[row_slice]
    provider = list(row["provider_risk_descriptor"])[row_slice]
    coordinate = list(row["provider_risk_coordinate"])[row_slice]
    return model.boundary_descriptor_projector_.transform(
        raw, provider, coordinate)


def fit_source_model(args, heldout, model_builder=None):
    model_builder = _fit_model_from_rows if model_builder is None else model_builder
    source_domains = [
        domain for domain in args.domains if domain != heldout]
    if len(source_domains) < 2:
        raise ValueError("TCB-V2 LODO needs at least two source domains")
    rows_by_domain = {
        domain: _source_rows(
            domain,
            count=args.source_records,
            replicates=args.source_replicates,
            source_seed=args.source_seed,
            d=args.d,
            L=args.L,
            sigma=args.sigma,
            alpha=args.alpha,
        )
        for domain in source_domains
    }
    model = model_builder(
        args, [rows_by_domain[domain] for domain in source_domains])
    return model, source_domains, rows_by_domain


def _pilot_indices(policy, model, descriptors, n0, rng):
    n0 = min(max(1, int(n0)), len(descriptors))
    if policy == "random":
        return np.asarray(rng.choice(
            len(descriptors), size=n0, replace=False), dtype=int)
    if policy != "source_boundary":
        raise ValueError(f"unknown pilot policy {policy!r}")
    prior = model.prior_adapter()
    source_margin = np.asarray(
        model.predict(descriptors, adapter=prior), dtype=float)
    boundary_count = min(n0, max(1, int(np.ceil(0.7 * n0))))
    selected = list(np.argsort(
        np.abs(source_margin), kind="stable")[:boundary_count])
    remaining = [index for index in range(len(descriptors)) if index not in selected]
    if len(selected) < n0:
        fill = rng.choice(
            remaining, size=n0 - len(selected), replace=False)
        selected.extend(int(index) for index in fill)
    return np.asarray(selected, dtype=int)


def _safe_spearman(truth, prediction):
    value = spearmanr(
        np.asarray(truth, dtype=float),
        np.asarray(prediction, dtype=float),
    ).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def _prediction_metrics(true_margin, mean, upper):
    true_margin = np.asarray(true_margin, dtype=float)
    mean = np.asarray(mean, dtype=float)
    upper = np.asarray(upper, dtype=float)
    true_safe = true_margin <= 0.0
    predicted_safe = upper <= 0.0
    false_safe = predicted_safe & ~true_safe
    recovered_safe = predicted_safe & true_safe
    boundary_count = min(
        len(true_margin), max(8, int(np.ceil(0.25 * len(true_margin)))))
    boundary = np.argsort(
        np.abs(true_margin), kind="stable")[:boundary_count]
    minimum_upper_index = int(np.argmin(upper))
    safest_truth_index = int(np.argmin(true_margin))
    return {
        "coverage_count": int(np.sum(true_margin <= upper)),
        "evaluation_count": int(len(true_margin)),
        "coverage_rate": float(np.mean(true_margin <= upper)),
        "predicted_safe_count": int(np.sum(predicted_safe)),
        "true_safe_count": int(np.sum(true_safe)),
        "false_safe_count": int(np.sum(false_safe)),
        "false_safe_rate": float(np.mean(false_safe)),
        "false_safe_conditional_rate": float(
            np.sum(false_safe) / max(int(np.sum(predicted_safe)), 1)),
        "safe_recall": float(
            np.sum(recovered_safe) / max(int(np.sum(true_safe)), 1)),
        "spearman": _safe_spearman(true_margin, mean),
        "margin_rmse": float(np.sqrt(np.mean((mean - true_margin) ** 2))),
        "boundary_mae": float(np.mean(np.abs(
            mean[boundary] - true_margin[boundary]))),
        "mean_upper_width": float(np.mean(upper - mean)),
        "minimum_true_margin": float(true_margin[safest_truth_index]),
        "minimum_predicted_mean": float(np.min(mean)),
        "minimum_predicted_upper": float(upper[minimum_upper_index]),
        "true_margin_at_minimum_predicted_upper": float(
            true_margin[minimum_upper_index]),
        "predicted_mean_at_safest_truth": float(mean[safest_truth_index]),
        "predicted_upper_at_safest_truth": float(upper[safest_truth_index]),
        "upper_width_at_safest_truth": float(
            upper[safest_truth_index] - mean[safest_truth_index]),
    }


def _aggregate_inner_metrics(folds):
    adapted = [fold["metrics"] for fold in folds]
    frozen = [fold["frozen_metrics"] for fold in folds]
    evaluation_count = sum(row["evaluation_count"] for row in adapted)
    coverage_count = sum(row["coverage_count"] for row in adapted)
    predicted_safe_count = sum(
        row["predicted_safe_count"] for row in adapted)
    false_safe_count = sum(row["false_safe_count"] for row in adapted)
    frozen_predicted_safe = sum(
        row["predicted_safe_count"] for row in frozen)
    frozen_false_safe = sum(row["false_safe_count"] for row in frozen)
    return {
        "fold_count": int(len(folds)),
        "folds": folds,
        "coverage_rate": float(
            coverage_count / max(evaluation_count, 1)),
        "predicted_safe_count": int(predicted_safe_count),
        "false_safe_count": int(false_safe_count),
        "false_safe_conditional_rate": float(
            false_safe_count / max(predicted_safe_count, 1)),
        "frozen_predicted_safe_count": int(frozen_predicted_safe),
        "frozen_false_safe_count": int(frozen_false_safe),
        "frozen_false_safe_conditional_rate": float(
            frozen_false_safe / max(frozen_predicted_safe, 1)),
        "mean_spearman": float(np.mean([
            row["spearman"] for row in adapted])),
        "frozen_mean_spearman": float(np.mean([
            row["spearman"] for row in frozen])),
        "rank_win_rate": float(np.mean([
            fold["rank_improved"] for fold in folds])),
        "nonvacuous_rate": float(np.mean([
            fold["nonvacuous_safe_set"] for fold in folds])),
        "target_domain_excluded_from_training_and_selection": True,
        "target_oracle_used": False,
    }


def source_inner_lodo(
    args,
    outer_heldout,
    source_domains,
    rows_by_domain,
    model_builder=None,
):
    """Selectability evidence using only domains other than outer target."""
    del outer_heldout
    model_builder = _fit_model_from_rows if model_builder is None else model_builder
    by_policy = {policy: [] for policy in args.pilot_policies}
    for inner_heldout in source_domains:
        training_domains = [
            domain for domain in source_domains if domain != inner_heldout]
        if len(training_domains) < 2:
            raise ValueError("nested source LODO needs two training domains")
        inner_model = model_builder(
            args,
            [rows_by_domain[domain] for domain in training_domains],
        )
        validation = rows_by_domain[inner_heldout]
        n_rows = len(validation["margin"])
        pilot_pool_count = min(
            max(4 * int(args.n0), int(args.n0)),
            max(int(args.n0), n_rows // 2),
        )
        projected = _project_row(inner_model, validation)
        pilot_descriptors = projected[:pilot_pool_count]
        evaluation_descriptors = projected[pilot_pool_count:]
        evaluation_margin = validation["margin"][pilot_pool_count:]
        prior_adapter = inner_model.prior_adapter()
        frozen_mean = inner_model.predict(
            evaluation_descriptors, adapter=prior_adapter)
        frozen_upper = inner_model.predict_upper(
            evaluation_descriptors, adapter=prior_adapter)
        frozen_metrics = _prediction_metrics(
            evaluation_margin, frozen_mean, frozen_upper)
        for policy in args.pilot_policies:
            rng = np.random.default_rng(_stable_seed(
                args.source_seed,
                f"nested:{inner_heldout}:{policy}:"
                f"{args.coordinate}:{args.geometry}:{args.rank}:"
                f"{args.adaptation_ridge}:{args.effect_ridge}",
            ))
            indices = _pilot_indices(
                policy,
                inner_model,
                pilot_descriptors,
                args.n0,
                rng,
            )
            adapter = inner_model.fit_target_adapter(
                pilot_descriptors[indices],
                validation["margin"][indices],
                pilot_variance=validation["variance"][indices],
                replicate_count=validation["replicate_count"][indices],
            )
            prediction = inner_model.predict(
                evaluation_descriptors, adapter=adapter)
            upper = inner_model.predict_upper(
                evaluation_descriptors, adapter=adapter)
            metrics = _prediction_metrics(
                evaluation_margin, prediction, upper)
            by_policy[policy].append({
                "inner_heldout": str(inner_heldout),
                "training_domains": list(training_domains),
                "metrics": metrics,
                "frozen_metrics": frozen_metrics,
                "rank_improved": bool(
                    metrics["spearman"]
                    > frozen_metrics["spearman"] + 1e-9),
                "false_safe_nonworse": bool(
                    metrics["false_safe_count"]
                    <= frozen_metrics["false_safe_count"]),
                "nonvacuous_safe_set": bool(
                    metrics["predicted_safe_count"] > 0),
                "adapter_effective_dimension": int(
                    adapter.effective_dimension),
                "adapter_diagnostics": adapter.diagnostics,
                "target_oracle_used": False,
            })
    return {
        policy: _aggregate_inner_metrics(folds)
        for policy, folds in by_policy.items()
    }


def run_gate(args, heldout, target_seed, model_builder=None):
    model, source_domains, rows_by_domain = fit_source_model(
        args, heldout, model_builder=model_builder)
    inner_lodo = source_inner_lodo(
        args,
        heldout,
        source_domains,
        rows_by_domain,
        model_builder=model_builder,
    )
    problem = _problem(
        heldout,
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
    )
    rng = np.random.default_rng(_stable_seed(target_seed, heldout))
    points = _unique_random_points(
        problem, args.pilot_pool + args.evaluation_pool, rng)
    pilot_points = points[:args.pilot_pool]
    evaluation_points = points[args.pilot_pool:]
    pilot_descriptors = np.vstack([
        LearnedMetaPrior.descriptor(problem, point)
        for point in pilot_points
    ])
    pilot_provider_descriptors = [
        LearnedMetaPrior.provider_risk_descriptor(problem, point)
        for point in pilot_points
    ]
    pilot_provider_coordinates = [
        LearnedMetaPrior.provider_risk_coordinate(problem, point)
        for point in pilot_points
    ]
    evaluation_descriptors = np.vstack([
        LearnedMetaPrior.descriptor(problem, point)
        for point in evaluation_points
    ])
    evaluation_provider_descriptors = [
        LearnedMetaPrior.provider_risk_descriptor(problem, point)
        for point in evaluation_points
    ]
    evaluation_provider_coordinates = [
        LearnedMetaPrior.provider_risk_coordinate(problem, point)
        for point in evaluation_points
    ]
    pilot_descriptors = model.boundary_descriptor_projector_.transform(
        pilot_descriptors,
        pilot_provider_descriptors,
        pilot_provider_coordinates,
    )
    evaluation_descriptors = model.boundary_descriptor_projector_.transform(
        evaluation_descriptors,
        evaluation_provider_descriptors,
        evaluation_provider_coordinates,
    )
    true_margin = np.asarray([
        float(problem.true_outputs(point)[1])
        + float(norm.ppf(1.0 - problem.alpha))
        * float(problem.true_sigma(point)[1])
        - float(problem.tau)
        for point in evaluation_points
    ], dtype=float)
    prior_adapter = model.prior_adapter()
    prior_mean = model.predict(
        evaluation_descriptors, adapter=prior_adapter)
    prior_upper = model.predict_upper(
        evaluation_descriptors, adapter=prior_adapter)
    prior_metrics = _prediction_metrics(
        true_margin, prior_mean, prior_upper)

    rows = []
    for policy in args.pilot_policies:
        policy_rng = np.random.default_rng(
            _stable_seed(target_seed, f"{heldout}:{policy}"))
        indices = _pilot_indices(
            policy, model, pilot_descriptors, args.n0, policy_rng)
        margins = []
        variances = []
        for index in indices:
            observed_margin, observed_variance = _replicate_margin(
                problem,
                pilot_points[int(index)],
                policy_rng,
                args.target_replicates,
            )
            margins.append(observed_margin)
            variances.append(observed_variance)
        adapter = model.fit_target_adapter(
            pilot_descriptors[indices],
            np.asarray(margins, dtype=float),
            pilot_variance=np.asarray(variances, dtype=float),
            replicate_count=np.full(
                len(indices), args.target_replicates, dtype=float),
        )
        prediction = model.predict(
            evaluation_descriptors, adapter=adapter)
        upper = model.predict_upper(
            evaluation_descriptors, adapter=adapter)
        metrics = _prediction_metrics(true_margin, prediction, upper)
        rows.append({
            "heldout": str(heldout),
            "target_seed": int(target_seed),
            "source_seed": int(args.source_seed),
            "source_domains": list(source_domains),
            "pilot_policy": str(policy),
            "n0": int(args.n0),
            "target_replicates": int(args.target_replicates),
            "target_evaluations_used": int(
                len(indices) * args.target_replicates),
            "coordinate": str(args.coordinate),
            "descriptor_mode": str(args.descriptor_mode),
            "provider_structural_input": bool(
                "provider_" in str(args.descriptor_mode)),
            "geometry": str(args.geometry),
            "rank": int(args.rank),
            "ridge": float(args.ridge),
            "domain_penalty": float(args.domain_penalty),
            "adaptation_ridge": float(args.adaptation_ridge),
            "effect_ridge": float(args.effect_ridge),
            "rotation_mode": str(args.rotation_mode),
            "rotation_ridge": float(args.rotation_ridge),
            "target_residual_rank": int(args.target_residual_rank),
            "residual_ridge": float(args.residual_ridge),
            "upper_alpha": float(args.upper_alpha),
            "adapter_mode": str(adapter.mode),
            "adapter_effective_dimension": int(adapter.effective_dimension),
            "adapter_diagnostics": adapter.diagnostics,
            "target_oracle_used_for_fit": False,
            "target_oracle_used_for_hyperparameter_selection": False,
            "evaluation_oracle_used_after_fit": True,
            "source_inner_lodo": inner_lodo[policy],
            "metrics": metrics,
            "frozen_metrics": prior_metrics,
            "rank_improved": bool(
                metrics["spearman"] > prior_metrics["spearman"] + 1e-9),
            "false_safe_nonworse": bool(
                metrics["false_safe_count"]
                <= prior_metrics["false_safe_count"]),
            "nonvacuous_safe_set": bool(
                metrics["predicted_safe_count"] > 0),
            **dict(getattr(model, "gate_row_metadata_", {})),
        })
    return {
        "schema_version": 1,
        "model": model.diagnostics(),
        "rows": rows,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--source-seed", type=int, default=7001)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--source-records", type=int, default=96)
    parser.add_argument("--source-replicates", type=int, default=3)
    parser.add_argument("--pilot-pool", type=int, default=256)
    parser.add_argument("--evaluation-pool", type=int, default=512)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--target-replicates", type=int, default=3)
    parser.add_argument(
        "--pilot-policies", default="random,source_boundary")
    parser.add_argument("--coordinate", default="boundary_latent")
    parser.add_argument(
        "--descriptor-mode",
        choices=tuple(sorted(LearnedMetaPrior.VALID_BOUNDARY_DESCRIPTOR_MODES)),
        default="learned_risk",
    )
    parser.add_argument("--meta-local-dim", type=int, default=3)
    parser.add_argument("--meta-shared-dim", type=int, default=2)
    parser.add_argument("--meta-soft-temperature", type=float, default=0.75)
    parser.add_argument("--geometry", default="low_rank_psd")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--domain-penalty", type=float, default=0.5)
    parser.add_argument("--boundary-temperature", type=float, default=1.0)
    parser.add_argument("--adaptation-ridge", type=float, default=5.0)
    parser.add_argument("--upper-alpha", type=float, default=0.01)
    parser.add_argument("--calibration-prior-df", type=float, default=2.0)
    parser.add_argument("--hierarchy-iterations", type=int, default=5)
    parser.add_argument("--effect-ridge", type=float, default=1.0)
    parser.add_argument(
        "--rotation-mode", choices=("none", "planar"), default="none")
    parser.add_argument("--rotation-ridge", type=float, default=5.0)
    parser.add_argument("--target-residual-rank", type=int, default=0)
    parser.add_argument("--residual-ridge", type=float, default=5.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.domains = tuple(parse_csv(args.domains))
    args.pilot_policies = tuple(parse_csv(args.pilot_policies))
    if args.heldout not in args.domains:
        raise ValueError("heldout domain must be present in --domains")
    result = run_gate(args, args.heldout, args.target_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(f"DONE tcb_v2_source_gate rows={len(result['rows'])}", flush=True)


if __name__ == "__main__":
    main()
