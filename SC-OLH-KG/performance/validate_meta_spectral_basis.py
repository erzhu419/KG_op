"""Validate the first LODO component before running a full KG benchmark.

The spectral basis is fit on source domains only.  A held-out target supplies a
small pilot set to fit ordinary ridge coefficients, then a disjoint test pool
measures whether the frozen representation improves sample efficiency over the
unfiltered coordinate basis.
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
from performance.benchmark_quality import json_safe, parse_csv, parse_weights, write_csv  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    LearnedMetaPrior,
    PilotGatedMetaPriorBasis,
)
from representation.adaptive_sparsity import AdaptiveSpikeSlabPosterior  # noqa: E402


def build_problem(name, args):
    return ScalarizedProblem(
        make_problem(
            name,
            d=args.d,
            L=args.L,
            sigma=args.sigma,
            alpha=args.alpha,
        ),
        weights=parse_weights(args.weights),
    )


def build_source_problems(names, args, seed):
    rng = np.random.default_rng(args.meta_seed + 4243 * int(seed))
    base_weights = parse_weights(args.weights)
    rows = []
    for name in names:
        for augment in range(max(1, int(args.source_augments))):
            if augment == 0:
                sigma = float(args.sigma)
                alpha = float(args.alpha)
                weights = base_weights
                label = name
            else:
                sigma = float(args.sigma) * float(np.exp(
                    rng.normal(0.0, max(args.source_sigma_jitter, 0.0))))
                alpha = float(args.alpha) * float(np.exp(
                    rng.normal(0.0, max(args.source_alpha_jitter, 0.0))))
                alpha = float(np.clip(alpha, 0.01, 0.20))
                weights = np.maximum(
                    base_weights + rng.normal(
                        0.0,
                        max(args.source_weight_jitter, 0.0),
                        size=len(base_weights),
                    ),
                    1e-4,
                )
                weights /= max(float(np.sum(weights)), 1e-12)
                label = f"{name}#aug{augment}"
            problem = ScalarizedProblem(
                make_problem(
                    name,
                    d=args.d,
                    L=args.L,
                    sigma=sigma,
                    alpha=alpha,
                ),
                weights=weights,
            )
            rows.append((label, problem))
    return rows


def coordinate_features(prior, problem, x):
    descriptor = prior._scaled_descriptor(prior.descriptor(problem, x))
    psi = prior.risk_coordinate(problem, x)
    cumulative = cumulative_feature_vector(prior.risk_exposure(problem, x))
    return np.concatenate([descriptor, psi, psi ** 2, cumulative[1:]])


def ridge_predict(train_x, train_y, test_x, ridge):
    x_mean = np.mean(train_x, axis=0)
    x_scale = np.std(train_x, axis=0)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    X = (train_x - x_mean) / x_scale
    X_test = (test_x - x_mean) / x_scale
    X = np.column_stack([np.ones(len(X)), X])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])
    y_mean = np.mean(train_y, axis=0)
    y_scale = np.std(train_y, axis=0)
    y_scale = np.where(y_scale < 1e-8, 1.0, y_scale)
    Y = (train_y - y_mean) / y_scale
    reg = float(ridge) * np.eye(X.shape[1], dtype=float)
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X.T @ X + reg, X.T @ Y, rcond=None)[0]
    return (X_test @ beta) * y_scale + y_mean


def ridge_loo_score(features, target, ridge):
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1, 1)
    errors = []
    for heldout in range(len(features)):
        train = np.arange(len(features)) != heldout
        prediction = ridge_predict(
            features[train],
            target[train],
            features[heldout:heldout + 1],
            ridge,
        )[0, 0]
        errors.append((float(target[heldout, 0]) - float(prediction)) ** 2)
    scale = float(np.var(target))
    return float(np.mean(errors) / max(scale, 1e-10))


def normalized_mse(truth, prediction):
    scale = float(np.var(truth))
    return float(np.mean((truth - prediction) ** 2) / max(scale, 1e-10))


def balanced_accuracy(truth, prediction):
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    scores = []
    for label in (False, True):
        mask = truth == label
        if np.any(mask):
            scores.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(scores)) if scores else 0.0


def decision_quality(problem, rows, truth, prediction):
    """Oracle-held-out metric used only to validate the learned gate."""

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    z_alpha = norm.ppf(1.0 - float(problem.alpha))
    sigma = np.asarray([
        float(problem.true_sigma(x)[1])
        if hasattr(problem, "true_sigma")
        else float(problem.sigma_level)
        for x in rows
    ], dtype=float)
    true_margin = truth[:, 1] + z_alpha * sigma - float(problem.tau)
    pred_margin = (
        prediction[:, 1]
        + z_alpha * float(problem.sigma_level)
        - float(problem.tau)
    )
    scale = max(
        float(problem.sigma_level),
        0.7413 * float(np.quantile(true_margin, 0.75) - np.quantile(true_margin, 0.25)),
        1e-8,
    )
    boundary_weight = 1.0 + 2.0 * np.exp(-0.5 * (true_margin / scale) ** 2)
    boundary_mse = float(np.average(
        ((true_margin - pred_margin) / scale) ** 2,
        weights=boundary_weight,
    ))
    true_feasible = true_margin <= 0.0
    pred_feasible = pred_margin <= 0.0
    false_feasible = (
        float(np.mean(~true_feasible[pred_feasible]))
        if np.any(pred_feasible)
        else 0.0
    )
    false_infeasible = (
        float(np.mean(~pred_feasible[true_feasible]))
        if np.any(true_feasible)
        else 0.0
    )
    if np.any(pred_feasible):
        eligible = np.where(pred_feasible)[0]
        selected = int(eligible[int(np.argmin(prediction[eligible, 0]))])
    else:
        selected = int(np.argmin(pred_margin))
    selected_violation = max(float(true_margin[selected]), 0.0) / scale
    selected_regret = 0.0
    if true_feasible[selected] and np.any(true_feasible):
        objective_scale = max(float(np.std(truth[:, 0])), 1e-8)
        selected_regret = max(
            float(truth[selected, 0]) - float(np.min(truth[true_feasible, 0])),
            0.0,
        ) / objective_scale
    elif np.any(true_feasible):
        selected_regret = 1.0
    total = (
        0.20 * boundary_mse
        + 3.0 * false_feasible
        + 0.25 * false_infeasible
        + selected_violation
        + 0.25 * selected_regret
    )
    return {
        "total": float(total),
        "boundary_mse": float(boundary_mse),
        "false_feasible_rate": float(false_feasible),
        "false_infeasible_rate": float(false_infeasible),
        "selected_true_feasible": bool(true_feasible[selected]),
        "selected_violation": float(selected_violation),
        "selected_regret": float(selected_regret),
    }


def target_rows(problem, n, rng):
    rows = []
    while len(rows) < n:
        rows.extend(problem.sample_random(rng) for _ in range(max(16, n - len(rows))))
        rows = unique_candidates(rows)
    return rows[:n]


def run_one(args, heldout, seed):
    domains = parse_csv(args.domains)
    source_names = [name for name in domains if name != heldout]
    sources = build_source_problems(source_names, args, seed)
    prior = LearnedMetaPrior(
        local_dim=args.meta_local_dim,
        shared_dim=args.meta_shared_dim,
        component_stage="spectral",
        spectral_active_dim=args.active_dim,
        spectral_max_library_size=args.max_library_size,
        spectral_low_frequency_components=args.low_frequency_components,
        spectral_graph_neighbors=args.graph_neighbors,
        spectral_relevance_floor=args.relevance_floor,
        spectral_gate_boundary_weight=args.gate_boundary_weight,
        spectral_gate_dangerous_weight=args.gate_dangerous_weight,
        spectral_gate_selection_tolerance=args.gate_selection_tolerance,
        spectral_gate_calibration_quantile=args.gate_calibration_quantile,
        spectral_frequency_adaptation=bool(args.frequency_adaptation),
        spectral_frequency_cutoffs=args.frequency_cutoffs,
        spectral_frequency_ridges=args.frequency_ridges,
        spectral_frequency_source_penalty=args.frequency_source_penalty,
        spectral_frequency_temperature=args.frequency_temperature,
        spectral_risk_alignment=bool(args.risk_alignment),
        spectral_alignment_active_dim=args.alignment_active_dim,
        spectral_alignment_subspace_dim=args.alignment_subspace_dim,
        spectral_alignment_domain_penalty=args.alignment_domain_penalty,
        spectral_alignment_source_procrustes=bool(
            args.alignment_source_procrustes),
        spectral_alignment_target_ridge=args.alignment_target_ridge,
        spectral_alignment_target_min_gain=args.alignment_target_min_gain,
        spectral_alignment_target_min_bins=args.alignment_target_min_bins,
        spectral_alignment_source_episodes=args.alignment_source_episodes,
        spectral_alignment_admission=bool(args.alignment_admission),
        spectral_alignment_episode_pilot_size=(
            args.alignment_episode_pilot_size),
        spectral_alignment_episode_evaluation_size=(
            args.alignment_episode_evaluation_size),
        spectral_alignment_episode_ridge=args.alignment_episode_ridge,
        spectral_additive_adaptation=bool(args.additive_adaptation),
        spectral_additive_max_groups=args.additive_max_groups,
        spectral_additive_target_max_groups=args.additive_target_max_groups,
        spectral_additive_source_penalty=args.additive_source_penalty,
        spectral_additive_complexity_penalty=args.additive_complexity_penalty,
        spectral_additive_temperature=args.additive_temperature,
        spectral_additive_refit_interval=args.additive_refit_interval,
        spectral_additive_max_saturation_fraction=(
            args.additive_max_saturation_fraction),
        spectral_adaptive_sparsity=bool(args.adaptive_sparsity),
        spectral_adaptive_spike_ratio=args.adaptive_spike_ratio,
        spectral_adaptive_residual_floor_scale=args.adaptive_residual_floor_scale,
        spectral_adaptive_gate_tolerance=args.adaptive_gate_tolerance,
        spectral_adaptive_multiplicity_correction=args.adaptive_multiplicity_correction,
        spectral_adaptive_max_effective_fraction=args.adaptive_max_effective_fraction,
        spectral_adaptive_saturation_fraction=args.adaptive_saturation_fraction,
        coordinate_mode=args.coordinate_mode,
        coordinate_relevance_floor=args.coordinate_relevance_floor,
        teacher_records_per_domain=args.source_boundary_records_per_domain,
        teacher_pool_size=args.source_boundary_pool_size,
        teacher_weight=args.source_boundary_weight,
        ridge=args.meta_ridge,
        seed=args.meta_seed + seed,
    ).fit_from_source_problems(
        sources,
        n_records_per_domain=args.source_records_per_domain,
        rng=np.random.default_rng(args.meta_seed + 1009 * seed),
    )

    target = build_problem(heldout, args)
    rng = np.random.default_rng(args.target_seed + 7919 * seed)
    rows = target_rows(target, args.pilot_size + args.test_size, rng)
    y = []
    observed_y = []
    z_alpha = norm.ppf(1.0 - float(target.alpha))
    for x in rows:
        true_y = target.true_outputs(x)
        y.append([float(true_y[0]), float(true_y[1])])
        observed = target.simulate(x, rng)
        observed_y.append([float(observed[0]), float(observed[1])])
    y = np.asarray(y, dtype=float)
    observed_y = np.asarray(observed_y, dtype=float)
    coordinate = np.vstack([coordinate_features(prior, target, x) for x in rows])
    psi = np.vstack([prior.risk_coordinate(target, x) for x in rows])
    fixed = psi[:, : min(int(args.active_dim), psi.shape[1])]
    spectral = np.vstack([prior.spectral_features(target, x) for x in rows])
    stage1_spectral = np.vstack([
        prior.stage1_spectral_features(target, x) for x in rows
    ])
    hybrid = np.hstack([fixed, spectral])
    split = int(args.pilot_size)
    coord_pred = ridge_predict(
        coordinate[:split], observed_y[:split], coordinate[split:], args.probe_ridge)
    fixed_pred = ridge_predict(
        fixed[:split], observed_y[:split], fixed[split:], args.probe_ridge)
    spectral_pred = ridge_predict(
        spectral[:split], observed_y[:split], spectral[split:], args.probe_ridge)
    hybrid_pred = ridge_predict(
        hybrid[:split], observed_y[:split], hybrid[split:], args.probe_ridge)
    representations = {
        "coordinate": coordinate,
        "fixed_psi": fixed,
        "source_spectral": stage1_spectral,
        "adaptive_spectral": spectral,
    }
    pilot_observations = {
        rows[index]: [observed_y[index].copy()]
        for index in range(split)
    }
    aligned_pred = None
    aligned_coordinate_pred = None
    frozen_aligned_coordinate_pred = None
    aligned_adapter = None
    if prior.risk_subspace_alignment is not None:
        aligned_adapter = prior.fit_target_risk_alignment(
            target, pilot_observations)
        aligned_coordinate = np.vstack([
            prior.risk_aligned_coordinate(
                target, x, adapter=aligned_adapter)
            for x in rows
        ])
        aligned = np.vstack([
            prior.risk_aligned_spectral_features(
                target, x, adapter=aligned_adapter)
            for x in rows
        ])
        representations["risk_aligned_coordinate"] = aligned_coordinate
        representations["risk_aligned_spectral"] = aligned
        frozen_aligned_coordinate = np.vstack([
            prior.frozen_risk_aligned_coordinate(target, x) for x in rows
        ])
        representations[
            "frozen_risk_aligned_coordinate"
        ] = frozen_aligned_coordinate
        aligned_coordinate_pred = ridge_predict(
            aligned_coordinate[:split], observed_y[:split],
            aligned_coordinate[split:], args.probe_ridge)
        aligned_pred = ridge_predict(
            aligned[:split], observed_y[:split], aligned[split:],
            args.probe_ridge)
        frozen_aligned_coordinate_pred = ridge_predict(
            frozen_aligned_coordinate[:split], observed_y[:split],
            frozen_aligned_coordinate[split:], args.probe_ridge)
        frozen_aligned_coordinate_pred[:, 1] += (
            prior.risk_subspace_alignment.source_residual_guard_
            * max(abs(float(target.tau)), float(target.sigma_level), 1e-6)
        )
    gated_pred = np.empty_like(spectral_pred)
    stage1_gated_pred = np.empty_like(spectral_pred)
    gate_selected = []
    gate_scores = {}
    gates = []
    for output_index in range(y.shape[1]):
        gate = PilotGatedMetaPriorBasis(
            prior,
            target,
            output_index=output_index,
            ridge=args.probe_ridge,
        )
        selected = gate.fit_from_observations(
            pilot_observations,
            output_index=output_index,
        )
        gate_diag = gate.diagnostics()
        stage1_selected = gate_diag.get("stage1_selected_basis", selected)
        stage1_matrix = representations.get(stage1_selected)
        if stage1_matrix is None:
            stage1_matrix = np.vstack([
                gate._variant_features(x, stage1_selected) for x in rows
            ])
        stage1_prediction = ridge_predict(
            stage1_matrix[:split],
            observed_y[:split, output_index:output_index + 1],
            stage1_matrix[split:],
            args.probe_ridge,
        )
        stage1_gated_pred[:, output_index] = stage1_prediction[:, 0]
        prediction_basis = (
            selected
            if (
                "frequency_band_" in str(selected)
                or str(selected).startswith("risk_aligned_")
                or selected == "frozen_risk_aligned_coordinate"
                or selected == "source_additive"
            )
            else stage1_selected
        )
        prediction_matrix = np.vstack([
            gate._variant_features(x, prediction_basis) for x in rows
        ])
        prediction_ridge = (
            float(gate_diag.get("selected_parametric_ridge", args.probe_ridge))
            if (
                "frequency_band_" in str(selected)
                or str(selected).startswith("risk_aligned_")
                or selected == "frozen_risk_aligned_coordinate"
                or selected == "source_additive"
            )
            else float(args.probe_ridge)
        )
        prediction = ridge_predict(
            prediction_matrix[:split],
            observed_y[:split, output_index:output_index + 1],
            prediction_matrix[split:],
            prediction_ridge,
        )
        if output_index == 1 and selected == "frozen_risk_aligned_coordinate":
            prediction[:, 0] += gate._risk_alignment_source_guard()
        gated_pred[:, output_index] = prediction[:, 0]
        gate_selected.append(selected)
        gate_scores[str(output_index)] = gate_diag
        gates.append(gate)
    adaptive_pred = gated_pred.copy()
    adaptive_enabled = False
    adaptive_diag = {"status": "disabled"}
    adaptive_spec = gates[1].adaptive_sparsity_spec(pilot_observations)
    if adaptive_spec is not None:
        posterior = AdaptiveSpikeSlabPosterior(
            adaptive_spec["source_pip"],
            adaptive_spec["source_slab_scale"],
            min_pip=adaptive_spec["min_pip"],
            max_pip=adaptive_spec["max_pip"],
            spike_ratio=adaptive_spec["spike_ratio"],
            damping=adaptive_spec["damping"],
            max_iter=adaptive_spec["max_iter"],
            tolerance=adaptive_spec["tolerance"],
            residual_floor_scale=adaptive_spec["residual_floor_scale"],
            multiplicity_correction=adaptive_spec["multiplicity_correction"],
            max_effective_fraction=adaptive_spec["max_effective_fraction"],
            always_active_count=adaptive_spec.get("always_active_count", 0),
            allowed_mask=adaptive_spec.get("allowed_mask"),
        ).fit(
            spectral[:split],
            observed_y[:split, 1],
            np.full(split, max(float(target.sigma_level) ** 2, 1e-6)),
            rows[:split],
            deviation_variance=max(float(target.sigma_level) ** 2, 1e-6),
        )
        adaptive_pred[:, 1] = posterior.predict_parametric_mean(spectral[split:])
        adaptive_enabled = True
        adaptive_diag = posterior.diagnostics()
    truth = y[split:]
    coord_nmse = [normalized_mse(truth[:, j], coord_pred[:, j]) for j in range(2)]
    fixed_nmse = [normalized_mse(truth[:, j], fixed_pred[:, j]) for j in range(2)]
    spectral_nmse = [normalized_mse(truth[:, j], spectral_pred[:, j]) for j in range(2)]
    hybrid_nmse = [normalized_mse(truth[:, j], hybrid_pred[:, j]) for j in range(2)]
    gated_nmse = [normalized_mse(truth[:, j], gated_pred[:, j]) for j in range(2)]
    adaptive_nmse = [
        normalized_mse(truth[:, j], adaptive_pred[:, j]) for j in range(2)
    ]
    frequency_enabled = any(
        "frequency_band_" in str(name) for name in gate_selected
    )
    alignment_enabled = any(
        str(name).startswith("risk_aligned_")
        or str(name).startswith("aligned_frequency_")
        or name == "frozen_risk_aligned_coordinate"
        for name in gate_selected
    )
    additive_enabled = any(name == "source_additive" for name in gate_selected)
    true_feasible = np.asarray([
        target.is_truly_feasible(x) for x in rows[split:]
    ], dtype=bool)
    coordinate_predicted_feasible = (
        coord_pred[:, 1] + z_alpha * float(target.sigma_level) <= float(target.tau)
    )
    spectral_predicted_feasible = (
        spectral_pred[:, 1] + z_alpha * float(target.sigma_level) <= float(target.tau)
    )
    decision = {
        "coordinate": decision_quality(target, rows[split:], truth, coord_pred),
        "fixed_psi": decision_quality(target, rows[split:], truth, fixed_pred),
        "source_spectral": decision_quality(
            target, rows[split:], truth, spectral_pred),
        "stage1_gated": decision_quality(
            target, rows[split:], truth, stage1_gated_pred),
        "gated": decision_quality(target, rows[split:], truth, gated_pred),
        "adaptive": decision_quality(target, rows[split:], truth, adaptive_pred),
    }
    if aligned_pred is not None:
        decision["frozen_risk_aligned_coordinate"] = decision_quality(
            target, rows[split:], truth, frozen_aligned_coordinate_pred)
        decision["risk_aligned_coordinate"] = decision_quality(
            target, rows[split:], truth, aligned_coordinate_pred)
        decision["risk_aligned_spectral"] = decision_quality(
            target, rows[split:], truth, aligned_pred)
    alignment_pilot_variant = None
    if aligned_pred is not None:
        if prior.alignment_episode_prior is not None and (
            prior.alignment_episode_prior.diagnostics().get("status") == "fit"
        ):
            alignment_pilot_variant = "frozen_risk_aligned_coordinate"
        else:
            alignment_pilot_variant = min(
                ("risk_aligned_coordinate", "risk_aligned_spectral"),
                key=lambda name: (
                    float(gate_scores["1"]["decision_score"][name]), name),
            )
    alignment_challenger_decision = (
        decision[alignment_pilot_variant]
        if alignment_pilot_variant is not None else None
    )
    alignment_deployed_decision = (
        decision["gated"]
        if alignment_enabled else decision["stage1_gated"]
    )
    diag = prior.spectral_basis.diagnostics()
    additive_diag = (
        prior.spectral_additive_bank.diagnostics()
        if prior.spectral_additive_bank is not None
        else {"max_offdiag_gram": 0.0, "max_diag_error": 0.0}
    )
    aligned_diag = (
        prior.risk_aligned_spectral_basis.diagnostics()
        if prior.risk_aligned_spectral_basis is not None
        else {"max_offdiag_gram": 0.0, "max_diag_error": 0.0}
    )
    aligned_additive_diag = (
        prior.risk_aligned_additive_bank.diagnostics()
        if prior.risk_aligned_additive_bank is not None
        else {"max_offdiag_gram": 0.0, "max_diag_error": 0.0}
    )
    return {
        "heldout": heldout,
        "source_domains": ",".join(source_names),
        "source_task_count": int(len(sources)),
        "seed": int(seed),
        "pilot_size": int(args.pilot_size),
        "test_size": int(args.test_size),
        "coordinate_dim": int(coordinate.shape[1]),
        "fixed_psi_dim": int(fixed.shape[1]),
        "spectral_dim": int(spectral.shape[1]),
        "hybrid_dim": int(hybrid.shape[1]),
        "coordinate_objective_nmse": coord_nmse[0],
        "spectral_objective_nmse": spectral_nmse[0],
        "objective_nmse_gain": coord_nmse[0] - spectral_nmse[0],
        "coordinate_constraint_nmse": coord_nmse[1],
        "spectral_constraint_nmse": spectral_nmse[1],
        "constraint_nmse_gain": coord_nmse[1] - spectral_nmse[1],
        "coordinate_mean_nmse": float(np.mean(coord_nmse)),
        "fixed_psi_mean_nmse": float(np.mean(fixed_nmse)),
        "spectral_mean_nmse": float(np.mean(spectral_nmse)),
        "hybrid_mean_nmse": float(np.mean(hybrid_nmse)),
        "gated_mean_nmse": float(np.mean(gated_nmse)),
        "adaptive_mean_nmse": float(np.mean(adaptive_nmse)),
        "adaptive_enabled": bool(adaptive_enabled),
        "frequency_enabled": bool(frequency_enabled),
        "frequency_over_stage1_decision_gain": (
            decision["stage1_gated"]["total"] - decision["gated"]["total"]
            if frequency_enabled else 0.0
        ),
        "alignment_enabled": bool(alignment_enabled),
        "alignment_adapter_accepted": bool(
            aligned_adapter is not None and aligned_adapter.accepted),
        "alignment_episode_prior_status": (
            prior.alignment_episode_diagnostics.get("status", "not_requested")
        ),
        "alignment_episode_count": int(
            prior.alignment_episode_diagnostics.get("n_episodes", 0)),
        "alignment_over_stage1_decision_gain": (
            decision["stage1_gated"]["total"]
            - alignment_deployed_decision["total"]
            if alignment_pilot_variant is not None else 0.0
        ),
        "alignment_pilot_variant": alignment_pilot_variant,
        "alignment_decision_loss": (
            alignment_deployed_decision["total"]
            if alignment_pilot_variant is not None else None
        ),
        "alignment_false_feasible_rate": (
            alignment_deployed_decision["false_feasible_rate"]
            if alignment_pilot_variant is not None else None
        ),
        "alignment_challenger_over_stage1_decision_gain": (
            decision["stage1_gated"]["total"]
            - alignment_challenger_decision["total"]
            if alignment_challenger_decision is not None else 0.0
        ),
        "alignment_challenger_decision_loss": (
            alignment_challenger_decision["total"]
            if alignment_challenger_decision is not None else None
        ),
        "alignment_challenger_false_feasible_rate": (
            alignment_challenger_decision["false_feasible_rate"]
            if alignment_challenger_decision is not None else None
        ),
        "alignment_coordinate_decision_loss": (
            decision["risk_aligned_coordinate"]["total"]
            if "risk_aligned_coordinate" in decision else None
        ),
        "alignment_spectral_decision_loss": (
            decision["risk_aligned_spectral"]["total"]
            if "risk_aligned_spectral" in decision else None
        ),
        "alignment_diagnostics": json.dumps(
            prior.risk_alignment_diagnostics, sort_keys=True),
        "target_alignment_diagnostics": json.dumps(
            aligned_adapter.diagnostics if aligned_adapter is not None else {},
            sort_keys=True,
        ),
        "additive_enabled": bool(additive_enabled),
        "additive_over_stage1_decision_gain": (
            decision["stage1_gated"]["total"] - decision["gated"]["total"]
            if additive_enabled else 0.0
        ),
        "adaptive_diagnostics": json.dumps(adaptive_diag, sort_keys=True),
        "adaptive_over_gated_nmse_gain": float(
            np.mean(gated_nmse) - np.mean(adaptive_nmse)),
        "mean_nmse_gain": float(np.mean(coord_nmse) - np.mean(spectral_nmse)),
        "hybrid_mean_nmse_gain": float(np.mean(coord_nmse) - np.mean(hybrid_nmse)),
        "hybrid_over_fixed_nmse_gain": float(np.mean(fixed_nmse) - np.mean(hybrid_nmse)),
        "gated_mean_nmse_gain": float(np.mean(coord_nmse) - np.mean(gated_nmse)),
        "gated_over_fixed_nmse_gain": float(np.mean(fixed_nmse) - np.mean(gated_nmse)),
        "gate_objective_basis": gate_selected[0],
        "gate_constraint_basis": gate_selected[1],
        "gate_scores": json.dumps(gate_scores, sort_keys=True),
        "coordinate_decision_loss": decision["coordinate"]["total"],
        "fixed_psi_decision_loss": decision["fixed_psi"]["total"],
        "spectral_decision_loss": decision["source_spectral"]["total"],
        "gated_decision_loss": decision["gated"]["total"],
        "adaptive_decision_loss": decision["adaptive"]["total"],
        "adaptive_over_gated_decision_gain": (
            decision["gated"]["total"] - decision["adaptive"]["total"]),
        "gated_over_fixed_decision_gain": (
            decision["fixed_psi"]["total"] - decision["gated"]["total"]),
        "gated_over_coordinate_decision_gain": (
            decision["coordinate"]["total"] - decision["gated"]["total"]),
        "coordinate_false_feasible_rate": decision[
            "coordinate"]["false_feasible_rate"],
        "fixed_psi_false_feasible_rate": decision[
            "fixed_psi"]["false_feasible_rate"],
        "spectral_false_feasible_rate": decision[
            "source_spectral"]["false_feasible_rate"],
        "stage1_false_feasible_rate": decision[
            "stage1_gated"]["false_feasible_rate"],
        "gated_false_feasible_rate": decision["gated"]["false_feasible_rate"],
        "adaptive_false_feasible_rate": decision[
            "adaptive"]["false_feasible_rate"],
        "coordinate_selected_true_feasible": decision[
            "coordinate"]["selected_true_feasible"],
        "fixed_psi_selected_true_feasible": decision[
            "fixed_psi"]["selected_true_feasible"],
        "gated_selected_true_feasible": decision[
            "gated"]["selected_true_feasible"],
        "learned_over_fixed_nmse_gain": float(
            np.mean(fixed_nmse) - np.mean(spectral_nmse)),
        "coordinate_feasible_balanced_accuracy": balanced_accuracy(
            true_feasible, coordinate_predicted_feasible),
        "spectral_feasible_balanced_accuracy": balanced_accuracy(
            true_feasible, spectral_predicted_feasible),
        "max_offdiag_gram": max(
            float(diag["max_offdiag_gram"]),
            float(additive_diag["max_offdiag_gram"]),
            float(aligned_diag["max_offdiag_gram"]),
            float(aligned_additive_diag["max_offdiag_gram"]),
        ),
        "max_diag_error": max(
            float(diag["max_diag_error"]),
            float(additive_diag["max_diag_error"]),
            float(aligned_diag["max_diag_error"]),
            float(aligned_additive_diag["max_diag_error"]),
        ),
        "selected_names": "|".join(diag["selected_names"]),
        "fingerprint": diag["fingerprint"],
    }


def summarize(rows):
    gains = np.asarray([row["mean_nmse_gain"] for row in rows], dtype=float)
    objective = np.asarray([row["objective_nmse_gain"] for row in rows], dtype=float)
    constraint = np.asarray([row["constraint_nmse_gain"] for row in rows], dtype=float)
    learned_over_fixed = np.asarray([
        row["learned_over_fixed_nmse_gain"] for row in rows
    ], dtype=float)
    hybrid_gains = np.asarray([
        row["hybrid_mean_nmse_gain"] for row in rows
    ], dtype=float)
    hybrid_over_fixed = np.asarray([
        row["hybrid_over_fixed_nmse_gain"] for row in rows
    ], dtype=float)
    gated_gains = np.asarray([
        row["gated_mean_nmse_gain"] for row in rows
    ], dtype=float)
    gated_over_fixed = np.asarray([
        row["gated_over_fixed_nmse_gain"] for row in rows
    ], dtype=float)
    decision_gain = np.asarray([
        row["gated_over_coordinate_decision_gain"] for row in rows
    ], dtype=float)
    adaptive_gain = np.asarray([
        row["adaptive_over_gated_decision_gain"] for row in rows
        if row["adaptive_enabled"]
    ], dtype=float)
    frequency_gain = np.asarray([
        row["frequency_over_stage1_decision_gain"] for row in rows
        if row.get("frequency_enabled", False)
    ], dtype=float)
    additive_gain = np.asarray([
        row["additive_over_stage1_decision_gain"] for row in rows
        if row.get("additive_enabled", False)
    ], dtype=float)
    alignment_gain = np.asarray([
        row["alignment_over_stage1_decision_gain"] for row in rows
        if row.get("alignment_decision_loss") is not None
    ], dtype=float)
    enabled_alignment_gain = np.asarray([
        row["alignment_over_stage1_decision_gain"] for row in rows
        if row.get("alignment_enabled", False)
    ], dtype=float)
    max_offdiag = max(float(row["max_offdiag_gram"]) for row in rows)
    by_heldout = {}
    for heldout in sorted({row["heldout"] for row in rows}):
        values = np.asarray([
            row["mean_nmse_gain"] for row in rows if row["heldout"] == heldout
        ], dtype=float)
        by_heldout[heldout] = {
            "median_mean_nmse_gain": float(np.median(values)),
            "win_rate": float(np.mean(values > 0.0)),
            "median_learned_over_fixed_nmse_gain": float(np.median([
                row["learned_over_fixed_nmse_gain"]
                for row in rows if row["heldout"] == heldout
            ])),
            "median_hybrid_mean_nmse_gain": float(np.median([
                row["hybrid_mean_nmse_gain"]
                for row in rows if row["heldout"] == heldout
            ])),
            "median_gated_mean_nmse_gain": float(np.median([
                row["gated_mean_nmse_gain"]
                for row in rows if row["heldout"] == heldout
            ])),
            "median_gated_over_coordinate_decision_gain": float(np.median([
                row["gated_over_coordinate_decision_gain"]
                for row in rows if row["heldout"] == heldout
            ])),
            "median_alignment_over_stage1_decision_gain": (
                float(np.median([
                    row["alignment_over_stage1_decision_gain"]
                    for row in rows
                    if row["heldout"] == heldout
                    and row.get("alignment_decision_loss") is not None
                ]))
                if any(
                    row["heldout"] == heldout
                    and row.get("alignment_decision_loss") is not None
                    for row in rows
                ) else None
            ),
            "median_alignment_false_feasible_rate": (
                float(np.median([
                    row["alignment_false_feasible_rate"]
                    for row in rows
                    if row["heldout"] == heldout
                    and row.get("alignment_false_feasible_rate") is not None
                ]))
                if any(
                    row["heldout"] == heldout
                    and row.get("alignment_false_feasible_rate") is not None
                    for row in rows
                ) else None
            ),
            "median_stage1_false_feasible_rate": float(np.median([
                row["stage1_false_feasible_rate"]
                for row in rows if row["heldout"] == heldout
            ])),
        }
    no_material_negative_transfer = all(
        value["median_gated_over_coordinate_decision_gain"] >= -0.05
        for value in by_heldout.values()
    )
    spectral_gate_rate = float(np.mean([
        row["gate_objective_basis"] in {"source_spectral", "adaptive_spectral"}
        or row["gate_constraint_basis"] in {"source_spectral", "adaptive_spectral"}
        or str(row["gate_objective_basis"]).startswith("frequency_band_")
        or str(row["gate_constraint_basis"]).startswith("frequency_band_")
        or str(row["gate_objective_basis"]).startswith("aligned_frequency_band_")
        or str(row["gate_constraint_basis"]).startswith("aligned_frequency_band_")
        or str(row["gate_objective_basis"]).startswith("risk_aligned_")
        or str(row["gate_constraint_basis"]).startswith("risk_aligned_")
        or row["gate_objective_basis"] == "source_additive"
        or row["gate_constraint_basis"] == "source_additive"
        for row in rows
    ]))
    frequency_enabled_rate = float(np.mean([
        row.get("frequency_enabled", False) for row in rows
    ]))
    accepted = bool(
        np.median(decision_gain) > 0.0
        and np.mean(decision_gain > 0.0) >= 0.60
        and np.median([
            row["gated_false_feasible_rate"] for row in rows
        ]) <= np.median([
            row["coordinate_false_feasible_rate"] for row in rows
        ])
        and spectral_gate_rate >= 0.20
        and no_material_negative_transfer
        and max_offdiag <= 1e-5
    )
    adaptive_enabled_rate = float(np.mean([
        row["adaptive_enabled"] for row in rows
    ]))
    adaptive_accepted = bool(
        len(adaptive_gain) > 0
        and np.median(adaptive_gain) >= 0.0
        and np.mean(adaptive_gain >= 0.0) >= 0.60
        and np.median([
            row["adaptive_false_feasible_rate"]
            for row in rows if row["adaptive_enabled"]
        ]) <= np.median([
            row["gated_false_feasible_rate"]
            for row in rows if row["adaptive_enabled"]
        ])
    )
    frequency_accepted = bool(
        len(frequency_gain) > 0
        and np.median(frequency_gain) >= 0.0
        and np.mean(frequency_gain >= 0.0) >= 0.60
        and np.median([
            row["gated_false_feasible_rate"] for row in rows
            if row.get("frequency_enabled", False)
        ]) <= np.median([
            row["stage1_false_feasible_rate"] for row in rows
            if row.get("frequency_enabled", False)
        ])
    )
    additive_accepted = bool(
        len(additive_gain) > 0
        and np.median(additive_gain) >= 0.0
        and np.mean(additive_gain >= 0.0) >= 0.60
        and np.median([
            row["gated_false_feasible_rate"] for row in rows
            if row.get("additive_enabled", False)
        ]) <= np.median([
            row["stage1_false_feasible_rate"] for row in rows
            if row.get("additive_enabled", False)
        ])
    )
    alignment_no_material_negative_transfer = all(
        value["median_alignment_over_stage1_decision_gain"] is not None
        and value["median_alignment_over_stage1_decision_gain"] >= -0.05
        for value in by_heldout.values()
    )
    alignment_false_feasible_no_worse = all(
        value["median_alignment_false_feasible_rate"] is not None
        and value["median_alignment_false_feasible_rate"]
        <= value["median_stage1_false_feasible_rate"] + 1e-12
        for value in by_heldout.values()
    )
    alignment_enabled_rate = float(np.mean([
        row.get("alignment_enabled", False) for row in rows
    ]))
    alignment_accepted = bool(
        len(enabled_alignment_gain) > 0
        and np.median(enabled_alignment_gain) > 0.0
        and np.mean(enabled_alignment_gain > 0.0) >= 0.60
        and np.median([
            row["alignment_false_feasible_rate"] for row in rows
            if row.get("alignment_enabled", False)
        ]) <= np.median([
            row["stage1_false_feasible_rate"] for row in rows
            if row.get("alignment_enabled", False)
        ])
        and alignment_enabled_rate >= 0.20
        and alignment_no_material_negative_transfer
        and alignment_false_feasible_no_worse
    )
    return {
        "accepted": accepted,
        "adaptive_accepted": adaptive_accepted,
        "frequency_accepted": frequency_accepted,
        "additive_accepted": additive_accepted,
        "alignment_accepted": alignment_accepted,
        "criterion": (
            "pilot-gated held-out decision-loss gain over the Stage-0 "
            "coordinate baseline > 0, win "
            "rate >= 0.60, no worse median false-feasible rate, learned "
            "spectral is selected in >= 20% of runs, every held-out median "
            "decision gain >= -0.05, "
            "max source Gram off-diagonal <= 1e-5"
        ),
        "n_runs": int(len(rows)),
        "median_mean_nmse_gain": float(np.median(gains)),
        "mean_nmse_win_rate": float(np.mean(gains > 0.0)),
        "median_objective_nmse_gain": float(np.median(objective)),
        "median_constraint_nmse_gain": float(np.median(constraint)),
        "median_learned_over_fixed_nmse_gain": float(np.median(learned_over_fixed)),
        "learned_over_fixed_win_rate": float(np.mean(learned_over_fixed > 0.0)),
        "median_hybrid_mean_nmse_gain": float(np.median(hybrid_gains)),
        "hybrid_mean_nmse_win_rate": float(np.mean(hybrid_gains > 0.0)),
        "median_hybrid_over_fixed_nmse_gain": float(np.median(hybrid_over_fixed)),
        "hybrid_over_fixed_win_rate": float(np.mean(hybrid_over_fixed > 0.0)),
        "median_gated_mean_nmse_gain": float(np.median(gated_gains)),
        "gated_mean_nmse_win_rate": float(np.mean(gated_gains > 0.0)),
        "median_gated_over_fixed_nmse_gain": float(np.median(gated_over_fixed)),
        "gated_over_fixed_win_rate": float(np.mean(gated_over_fixed > 0.0)),
        "median_gated_over_coordinate_decision_gain": float(np.median(decision_gain)),
        "gated_over_coordinate_decision_win_rate": float(
            np.mean(decision_gain > 0.0)),
        "median_coordinate_false_feasible_rate": float(np.median([
            row["coordinate_false_feasible_rate"] for row in rows
        ])),
        "median_fixed_psi_false_feasible_rate": float(np.median([
            row["fixed_psi_false_feasible_rate"] for row in rows
        ])),
        "median_gated_false_feasible_rate": float(np.median([
            row["gated_false_feasible_rate"] for row in rows
        ])),
        "spectral_gate_rate": spectral_gate_rate,
        "frequency_enabled_rate": frequency_enabled_rate,
        "alignment_enabled_rate": alignment_enabled_rate,
        "alignment_adapter_accepted_rate": float(np.mean([
            row.get("alignment_adapter_accepted", False) for row in rows
        ])),
        "median_alignment_over_stage1_decision_gain": (
            float(np.median(alignment_gain)) if len(alignment_gain) else None
        ),
        "alignment_over_stage1_decision_win_rate": (
            float(np.mean(alignment_gain > 0.0))
            if len(alignment_gain) else None
        ),
        "median_enabled_alignment_over_stage1_decision_gain": (
            float(np.median(enabled_alignment_gain))
            if len(enabled_alignment_gain) else None
        ),
        "enabled_alignment_over_stage1_decision_win_rate": (
            float(np.mean(enabled_alignment_gain > 0.0))
            if len(enabled_alignment_gain) else None
        ),
        "alignment_no_material_negative_transfer": bool(
            alignment_no_material_negative_transfer),
        "alignment_false_feasible_no_worse_by_domain": bool(
            alignment_false_feasible_no_worse),
        "median_frequency_over_stage1_decision_gain": (
            float(np.median(frequency_gain)) if len(frequency_gain) else None
        ),
        "frequency_over_stage1_decision_win_rate": (
            float(np.mean(frequency_gain >= 0.0)) if len(frequency_gain) else None
        ),
        "additive_enabled_rate": float(np.mean([
            row.get("additive_enabled", False) for row in rows
        ])),
        "median_additive_over_stage1_decision_gain": (
            float(np.median(additive_gain)) if len(additive_gain) else None
        ),
        "additive_over_stage1_decision_win_rate": (
            float(np.mean(additive_gain >= 0.0)) if len(additive_gain) else None
        ),
        "adaptive_enabled_rate": adaptive_enabled_rate,
        "median_adaptive_over_gated_decision_gain": (
            float(np.median(adaptive_gain)) if len(adaptive_gain) else None),
        "adaptive_over_gated_decision_win_rate": (
            float(np.mean(adaptive_gain >= 0.0)) if len(adaptive_gain) else None),
        "median_adaptive_false_feasible_rate": float(np.median([
            row["adaptive_false_feasible_rate"] for row in rows
        ])),
        "no_material_negative_transfer": bool(no_material_negative_transfer),
        "max_offdiag_gram": max_offdiag,
        "by_heldout": by_heldout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl",
    )
    parser.add_argument("--heldouts", default="")
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--source_records_per_domain", type=int, default=96)
    parser.add_argument(
        "--source_boundary_records_per_domain", type=int, default=0)
    parser.add_argument("--source_boundary_pool_size", type=int, default=512)
    parser.add_argument("--source_boundary_weight", type=float, default=3.0)
    parser.add_argument("--source_augments", type=int, default=1)
    parser.add_argument("--source_sigma_jitter", type=float, default=0.20)
    parser.add_argument("--source_alpha_jitter", type=float, default=0.25)
    parser.add_argument("--source_weight_jitter", type=float, default=0.05)
    parser.add_argument("--pilot_size", type=int, default=12)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--meta_local_dim", type=int, default=3)
    parser.add_argument("--meta_shared_dim", type=int, default=3)
    parser.add_argument("--active_dim", type=int, default=6)
    parser.add_argument("--max_library_size", type=int, default=64)
    parser.add_argument("--low_frequency_components", type=int, default=8)
    parser.add_argument("--graph_neighbors", type=int, default=10)
    parser.add_argument("--relevance_floor", type=float, default=0.05)
    parser.add_argument("--gate_boundary_weight", type=float, default=2.0)
    parser.add_argument("--gate_dangerous_weight", type=float, default=3.0)
    parser.add_argument("--gate_selection_tolerance", type=float, default=0.02)
    parser.add_argument("--gate_calibration_quantile", type=float, default=0.90)
    parser.add_argument(
        "--frequency_adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--frequency_cutoffs", default="3,5,8,12")
    parser.add_argument("--frequency_ridges", default="0.0001,0.01,1.0")
    parser.add_argument("--frequency_source_penalty", type=float, default=0.05)
    parser.add_argument("--frequency_temperature", type=float, default=0.5)
    parser.add_argument(
        "--risk_alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--alignment_active_dim", type=int, default=4)
    parser.add_argument("--alignment_subspace_dim", type=int, default=2)
    parser.add_argument("--alignment_domain_penalty", type=float, default=0.5)
    parser.add_argument(
        "--alignment_source_procrustes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--alignment_target_ridge", type=float, default=5.0)
    parser.add_argument("--alignment_target_min_gain", type=float, default=0.02)
    parser.add_argument("--alignment_target_min_bins", type=int, default=3)
    parser.add_argument("--alignment_source_episodes", type=int, default=0)
    parser.add_argument(
        "--alignment_admission",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--alignment_episode_pilot_size", type=int, default=10)
    parser.add_argument(
        "--alignment_episode_evaluation_size", type=int, default=24)
    parser.add_argument("--alignment_episode_ridge", type=float, default=0.1)
    parser.add_argument(
        "--additive_adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--additive_max_groups", type=int, default=8)
    parser.add_argument("--additive_target_max_groups", type=int, default=2)
    parser.add_argument("--additive_source_penalty", type=float, default=0.05)
    parser.add_argument(
        "--additive_complexity_penalty", type=float, default=0.05)
    parser.add_argument("--additive_temperature", type=float, default=0.5)
    parser.add_argument("--additive_refit_interval", type=int, default=5)
    parser.add_argument(
        "--additive_max_saturation_fraction", type=float, default=0.20)
    parser.add_argument(
        "--adaptive_sparsity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--adaptive_spike_ratio", type=float, default=0.05)
    parser.add_argument(
        "--adaptive_residual_floor_scale", type=float, default=0.05)
    parser.add_argument("--adaptive_gate_tolerance", type=float, default=0.05)
    parser.add_argument(
        "--adaptive_multiplicity_correction", type=float, default=1.0)
    parser.add_argument(
        "--adaptive_max_effective_fraction", type=float, default=0.35)
    parser.add_argument(
        "--adaptive_saturation_fraction", type=float, default=0.90)
    parser.add_argument(
        "--coordinate_mode",
        choices=["pca", "stable_supervised"],
        default="stable_supervised",
    )
    parser.add_argument("--coordinate_relevance_floor", type=float, default=0.05)
    parser.add_argument("--meta_ridge", type=float, default=1e-4)
    parser.add_argument("--probe_ridge", type=float, default=0.1)
    parser.add_argument("--meta_seed", type=int, default=20260710)
    parser.add_argument("--target_seed", type=int, default=20260711)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="meta_spectral_validation")
    args = parser.parse_args()

    heldouts = parse_csv(args.heldouts) or parse_csv(args.domains)
    rows = [
        run_one(args, heldout, seed)
        for heldout in heldouts
        for seed in range(args.seed_start, args.seed_start + args.n_seeds)
    ]
    summary = summarize(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{args.out_prefix}_rows.csv"
    json_path = out_dir / f"{args.out_prefix}.json"
    write_csv(rows_path, rows)
    json_path.write_text(json.dumps(json_safe({
        "config": vars(args),
        "summary": summary,
        "rows": rows,
    }), indent=2), encoding="utf-8")
    print(json.dumps(json_safe({
        "summary": summary,
        "rows_csv": str(rows_path),
        "json": str(json_path),
    }), indent=2))


if __name__ == "__main__":
    main()
