"""Summarize one-seed manifest shards for the LODO promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _median(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.median(values))


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _input_files(inputs):
    files = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("result.json")))
        elif path.is_file():
            files.append(path)
    return files


def load_rows(inputs):
    rows = []
    seen = set()
    for path in _input_files(inputs):
        payload = json.loads(path.read_text())
        if len(payload.get("rows", [])) != 1:
            raise ValueError(f"one-seed shard expected in {path}")
        row = dict(payload["rows"][0])
        overrides = dict(payload.get("causal_overrides", {}))
        mode = str(overrides.get(
            "exact_kg_sampling_mode", row.get("exact_kg_sampling_mode", "iid")))
        ordered = bool(overrides.get("meta_ordered_cumulative_exposure", False))
        basis_mode = str(overrides.get(
            "meta_ordered_exposure_basis_mode", "full_quadratic"))
        adaptive = bool(overrides.get(
            "meta_ordered_exposure_adaptive_sparsity", False))
        replace = bool(overrides.get(
            "meta_ordered_exposure_replace_local_kernel", False))
        semiparametric = bool(overrides.get(
            "meta_ordered_exposure_semiparametric_residual", False))
        latent_structure = bool(overrides.get(
            "meta_ordered_exposure_latent_structure_selection", False))
        group_shared = bool(overrides.get(
            "meta_ordered_exposure_group_shared_shrinkage", False))
        group_ridge = bool(overrides.get(
            "meta_ordered_exposure_group_ridge_learning", False))
        safe_generalized = bool(overrides.get(
            "task_posterior_safe_generalized", False))
        finalist_budget = int(overrides.get(
            "finalist_replication_budget", 0) or 0)
        finalist_expert_stratified = bool(overrides.get(
            "finalist_replication_expert_stratified", False))
        finalist_adaptive_race = bool(overrides.get(
            "finalist_replication_adaptive_race", False))
        finalist_fixed_universe = bool(overrides.get(
            "finalist_replication_fixed_universe", False))
        if ordered and (
            basis_mode != "full_quadratic" or adaptive or replace
        ):
            basis_label = (
                "diag" if basis_mode == "diagonal_quadratic" else "full")
            cell_parts = ["ordered"]
            if semiparametric:
                cell_parts.append("semiparametric")
            if latent_structure:
                cell_parts.append("latent_structure")
            if group_shared:
                cell_parts.append("group_shared")
            if group_ridge:
                cell_parts.append("group_ridge")
            if safe_generalized:
                cell_parts.append("safe_generalized")
            if finalist_budget > 0:
                cell_parts.append(f"finalist_r{finalist_budget}")
            if finalist_expert_stratified:
                cell_parts.append("expert_stratified")
            if finalist_adaptive_race:
                cell_parts.append("adaptive_race")
            if finalist_fixed_universe:
                cell_parts.append("fixed_universe")
            cell = "_".join(cell_parts + [
                basis_label,
                "sparse" if adaptive else "dense",
                "replace" if replace else "add",
                mode,
            ])
        else:
            cell = f"{'ordered' if ordered else 'baseline'}_{mode}"
        key = (cell, str(row["heldout"]), int(row["seed"]))
        if key in seen:
            raise ValueError(f"duplicate shard {key}")
        seen.add(key)
        row["_cell"] = cell
        row["_path"] = str(path)
        rows.append(row)
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["_cell"], str(row["heldout"])), []).append(row)
    summaries = []
    for (cell, heldout), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["seed"]))
        ordered_weights = []
        ordered_constraint_dimensions = []
        ordered_constraint_dimension_caps = []
        ordered_dimension_budget_respected = []
        ordered_local_inclusion_mass = []
        ordered_projection_error = []
        ordered_curvature_group_pip = []
        ordered_shared_group_pip = []
        ordered_complexity_selection_valid = []
        ordered_linear_group_penalty = []
        ordered_curvature_group_penalty = []
        ordered_shared_group_penalty = []
        selected_frequencies = []
        safe_generalized = []
        safe_history_counts = []
        safe_pairwise_pairs = []
        safe_pairwise_effective_weights = []
        predictive_safe_total_variation = []
        safe_effective_experts = []
        predictive_ordered_weights = []
        safe_ordered_weights = []
        finalist_forced_evaluations = []
        finalist_target_counts = []
        finalist_min_replicate_satisfied = []
        finalist_adaptive_race = []
        finalist_refresh_counts = []
        finalist_completed_target_counts = []
        finalist_fixed_universe = []
        finalist_fixed_universe_sizes = []
        finalist_incomplete_archive_counts = []
        for row in group:
            task_posterior = row.get("task_posterior") or {}
            posterior = task_posterior.get("posterior") or {}
            by_expert = posterior.get("decision_by_expert", {})
            ordered_name = (
                "ordered_semiparametric"
                if "ordered_semiparametric" in by_expert
                else "ordered_cumulative"
            )
            ordered_weights.append(by_expert.get(ordered_name))
            safe_generalized.append(bool(
                posterior.get("safe_generalized", False)))
            safe_history_counts.append(
                task_posterior.get("safe_history_count"))
            last_update = posterior.get("last_update") or {}
            safe_pairwise_pairs.append(
                last_update.get("safe_pairwise_pairs"))
            safe_pairwise_effective_weights.append(
                last_update.get("safe_pairwise_effective_weight"))
            safe_effective_experts.append(
                posterior.get("safe_effective_experts"))
            predictive_by_expert = posterior.get(
                "posterior_by_expert", {}) or {}
            safe_by_expert = posterior.get(
                "safe_posterior_by_expert", {}) or {}
            predictive_ordered_weights.append(
                predictive_by_expert.get(ordered_name))
            safe_ordered_weights.append(
                safe_by_expert.get(ordered_name))
            predictive_weights = np.asarray(
                posterior.get("posterior_weights", []), dtype=float)
            safe_weights = np.asarray(
                posterior.get("safe_posterior_weights", []), dtype=float)
            if (
                len(predictive_weights) > 0
                and predictive_weights.shape == safe_weights.shape
                and np.all(np.isfinite(predictive_weights))
                and np.all(np.isfinite(safe_weights))
            ):
                predictive_safe_total_variation.append(float(
                    0.5 * np.sum(np.abs(
                        predictive_weights - safe_weights))))
            finalist = row.get("finalist_replication") or {}
            finalist_adaptive_race.append(bool(
                finalist.get("adaptive_race", False)))
            finalist_fixed_universe.append(bool(
                finalist.get("fixed_universe", False)))
            finalist_fixed_universe_sizes.append(
                finalist.get("fixed_universe_size"))
            finalist_refresh_counts.append(len(
                finalist.get("refresh_history") or []))
            finalist_completed_target_counts.append(
                finalist.get("completed_target_count"))
            finalist_forced_evaluations.append(
                finalist.get("forced_evaluations"))
            targets = finalist.get("targets")
            if targets is not None:
                finalist_target_counts.append(len(targets))
            replicate_counts = finalist.get("replicate_counts") or []
            minimum_replicates = finalist.get("minimum_replicates")
            if replicate_counts and minimum_replicates is not None:
                completed_count = int(sum(
                    int(value) >= int(minimum_replicates)
                    for value in replicate_counts
                ))
                finalist_incomplete_archive_counts.append(
                    len(replicate_counts) - completed_count)
                finalist_min_replicate_satisfied.append(
                    completed_count > 0
                    if bool(finalist.get("adaptive_race", False))
                    else completed_count == len(replicate_counts)
                )
            ordered_expert = next((
                expert
                for expert in (row.get("task_posterior") or {}).get(
                    "experts", [])
                if expert.get("name") == ordered_name
            ), None)
            if ordered_expert is not None:
                adaptive = ordered_expert.get(
                    "gpr_adaptive_sparsity", [])
                if len(adaptive) > 1:
                    ordered_constraint_dimensions.append(
                        adaptive[1].get("effective_dimension"))
                    ordered_constraint_dimension_caps.append(
                        adaptive[1].get("max_effective_dimension"))
                    effective = adaptive[1].get("effective_dimension")
                    cap = adaptive[1].get("max_effective_dimension")
                    if effective is not None and cap is not None:
                        ordered_dimension_budget_respected.append(
                            float(effective) <= float(cap) + 1e-10)
                    projection = (ordered_expert.get("basis") or {}).get(
                        "ordered_residual_projection") or {}
                    residual_dim = int(projection.get("residual_dim", 0))
                    posterior_pip = adaptive[1].get("posterior_pip", [])
                    if residual_dim and len(posterior_pip) >= residual_dim:
                        ordered_local_inclusion_mass.append(sum(
                            posterior_pip[-residual_dim:]))
                    ordered_projection_error.append(
                        projection.get("orthogonality_relative"))
                    shared_groups = adaptive[1].get(
                        "shared_shrinkage_groups", {}) or {}
                    curvature = shared_groups.get("0", {}) or {}
                    shared = shared_groups.get("1", {}) or {}
                    ordered_curvature_group_pip.append(
                        curvature.get("posterior_pip"))
                    ordered_shared_group_pip.append(
                        shared.get("posterior_pip"))
                    if adaptive[1].get("method") == "nested_loo_group_ridge":
                        ordered_complexity_selection_valid.append(bool(
                            adaptive[1].get("complexity_selection_valid")))
                        ridge_groups = adaptive[1].get("groups", {}) or {}
                        ordered_linear_group_penalty.append(
                            (ridge_groups.get("0", {}) or {}).get(
                                "selected_penalty"))
                        ordered_curvature_group_penalty.append(
                            (ridge_groups.get("1", {}) or {}).get(
                                "selected_penalty"))
                        ordered_shared_group_penalty.append(
                            (ridge_groups.get("2", {}) or {}).get(
                                "selected_penalty"))
            ordered_diag = (
                (row.get("meta_prior") or {}).get(
                    "ordered_cumulative_exposure") or {})
            frequencies = ordered_diag.get("selected_frequencies")
            if frequencies is not None:
                selected_frequencies.append(tuple(map(int, frequencies)))
        frequency_counts = {}
        for frequencies in selected_frequencies:
            key = ",".join(map(str, frequencies))
            frequency_counts[key] = frequency_counts.get(key, 0) + 1
        summaries.append({
            "cell": cell,
            "heldout": heldout,
            "n_seeds": len(group),
            "seeds": [int(row["seed"]) for row in group],
            "true_feasible_count": int(sum(
                bool(row["true_feasible"]) for row in group)),
            "posterior_certified_count": int(sum(
                bool(row["posterior_feasible"]) for row in group)),
            "false_feasible_count": int(sum(
                bool(row["false_feasible"]) for row in group)),
            "initial_has_true_feasible_count": int(sum(
                bool(row.get("initial_has_true_feasible")) for row in group)),
            "median_feasible_regret": _median(
                row.get("feasible_simple_regret") for row in group),
            "mean_constraint_violation": _mean(
                row.get("constraint_violation") for row in group),
            "median_true_chance_margin": _median(
                row.get("true_chance_margin") for row in group),
            "median_algorithm_time_sec": _median(
                row.get("algorithm_time_sec") for row in group),
            "median_wall_time_sec": _median(
                row.get("wall_time_sec") for row in group),
            "ordered_expert_weight_median": _median(ordered_weights),
            "safe_generalized_count": int(sum(safe_generalized)),
            "safe_history_count_median": _median(safe_history_counts),
            "safe_pairwise_pairs_median": _median(safe_pairwise_pairs),
            "safe_pairwise_effective_weight_median": _median(
                safe_pairwise_effective_weights),
            "predictive_safe_total_variation_median": _median(
                predictive_safe_total_variation),
            "safe_effective_experts_median": _median(
                safe_effective_experts),
            "predictive_ordered_expert_weight_median": _median(
                predictive_ordered_weights),
            "safe_ordered_expert_weight_median": _median(
                safe_ordered_weights),
            "finalist_forced_evaluations_median": _median(
                finalist_forced_evaluations),
            "finalist_target_count_median": _median(
                finalist_target_counts),
            "finalist_min_replicate_checked_count": int(
                len(finalist_min_replicate_satisfied)),
            "finalist_min_replicate_failure_count": int(sum(
                not value for value in finalist_min_replicate_satisfied)),
            "finalist_adaptive_race_count": int(sum(
                finalist_adaptive_race)),
            "finalist_refresh_count_median": _median(
                finalist_refresh_counts),
            "finalist_completed_target_count_median": _median(
                finalist_completed_target_counts),
            "finalist_fixed_universe_count": int(sum(
                finalist_fixed_universe)),
            "finalist_fixed_universe_size_median": _median(
                finalist_fixed_universe_sizes),
            "finalist_incomplete_archive_target_count_median": _median(
                finalist_incomplete_archive_counts),
            "ordered_constraint_effective_dimension_median": _median(
                ordered_constraint_dimensions),
            "ordered_constraint_dimension_cap_median": _median(
                ordered_constraint_dimension_caps),
            "ordered_dimension_budget_checked_count": int(
                len(ordered_dimension_budget_respected)),
            "ordered_dimension_budget_violation_count": int(sum(
                not value for value in ordered_dimension_budget_respected)),
            "ordered_local_inclusion_mass_median": _median(
                ordered_local_inclusion_mass),
            "ordered_projection_error_median": _median(
                ordered_projection_error),
            "ordered_curvature_group_pip_median": _median(
                ordered_curvature_group_pip),
            "ordered_shared_group_pip_median": _median(
                ordered_shared_group_pip),
            "ordered_complexity_selection_checked_count": int(
                len(ordered_complexity_selection_valid)),
            "ordered_complexity_selection_invalid_count": int(sum(
                not value for value in ordered_complexity_selection_valid)),
            "ordered_linear_group_penalty_median": _median(
                ordered_linear_group_penalty),
            "ordered_curvature_group_penalty_median": _median(
                ordered_curvature_group_penalty),
            "ordered_shared_group_penalty_median": _median(
                ordered_shared_group_penalty),
            "ordered_frequency_counts": frequency_counts,
        })

    by_cell = {}
    for row in summaries:
        by_cell.setdefault(row["cell"], {})[row["heldout"]] = row
    gates = []
    for cell, domains in sorted(by_cell.items()):
        factor = domains.get("FactorShockStatePolicyRZDT1")
        inventory = domains.get("InventorySupplyChain")
        complete = bool(
            factor is not None and inventory is not None
            and factor["n_seeds"] == 7 and inventory["n_seeds"] == 7)
        if "group_ridge" in cell:
            complexity_valid = bool(
                factor is not None and inventory is not None
                and factor[
                    "ordered_complexity_selection_checked_count"] == 7
                and inventory[
                    "ordered_complexity_selection_checked_count"] == 7
                and factor[
                    "ordered_complexity_selection_invalid_count"] == 0
                and inventory[
                    "ordered_complexity_selection_invalid_count"] == 0
            )
        else:
            complexity_valid = bool(
                factor is not None and inventory is not None
                and factor["ordered_dimension_budget_checked_count"] == 7
                and inventory["ordered_dimension_budget_checked_count"] == 7
                and factor["ordered_dimension_budget_violation_count"] == 0
                and inventory["ordered_dimension_budget_violation_count"] == 0
            )
        passed = bool(
            complete
            and factor["true_feasible_count"] == 7
            and factor["mean_constraint_violation"] <= 1e-12
            and inventory["true_feasible_count"] >= 4
            and inventory["false_feasible_count"] <= 1
            and complexity_valid
        )
        gates.append({
            "cell": cell,
            "complete": complete,
            "passed": passed,
            "factor_true_feasible": (
                None if factor is None else factor["true_feasible_count"]),
            "inventory_true_feasible": (
                None if inventory is None else inventory["true_feasible_count"]),
            "inventory_false_feasible": (
                None if inventory is None else inventory["false_feasible_count"]),
            "factor_dimension_budget_violations": (
                None if factor is None else factor[
                    "ordered_dimension_budget_violation_count"]),
            "inventory_dimension_budget_violations": (
                None if inventory is None else inventory[
                    "ordered_dimension_budget_violation_count"]),
            "factor_complexity_selection_invalid": (
                None if factor is None else factor[
                    "ordered_complexity_selection_invalid_count"]),
            "inventory_complexity_selection_invalid": (
                None if inventory is None else inventory[
                    "ordered_complexity_selection_invalid_count"]),
        })
    return {"schema_version": 1, "summaries": summaries, "gates": gates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = summarize(load_rows(args.inputs))
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
