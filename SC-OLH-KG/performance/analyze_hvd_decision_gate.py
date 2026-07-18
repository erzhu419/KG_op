#!/usr/bin/env python3
"""Aggregate the frozen-proposal cumulative-HVD causal closure gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


FACTORS = {
    "hvd": ("pooled", "factor_cumulative"),
    "discrepancy": ("frozen", "adaptive"),
    "action": ("new_only", "hvd_voi"),
    "certificate": ("separable", "joint_tangent"),
    "mean_profile": ("eta_source_adaptive", "eta_source_sequential"),
}


def _factor_levels(rows):
    levels = {}
    hvd = {str(row.get("hvd")) for row in rows}
    if {"factor_cumulative", "factor_hierarchical"}.issubset(hvd):
        levels["hvd"] = ("factor_cumulative", "factor_hierarchical")
    elif {"pooled", "factor_cumulative"}.issubset(hvd):
        levels["hvd"] = ("pooled", "factor_cumulative")

    discrepancy = {str(row.get("discrepancy")) for row in rows}
    if {"frozen", "adaptive"}.issubset(discrepancy):
        levels["discrepancy"] = ("frozen", "adaptive")

    actions = {str(row.get("action")) for row in rows}
    if {"new_only", "joint_voi"}.issubset(actions):
        levels["action"] = ("new_only", "joint_voi")
    elif {"new_only", "hvd_voi"}.issubset(actions):
        levels["action"] = ("new_only", "hvd_voi")

    certificates = {str(row.get("certificate")) for row in rows}
    if {"separable", "joint_tangent"}.issubset(certificates):
        levels["certificate"] = ("separable", "joint_tangent")

    mean_profiles = {str(row.get("mean_profile")) for row in rows}
    if {"eta_source_adaptive", "eta_source_sequential"}.issubset(
        mean_profiles
    ):
        levels["mean_profile"] = (
            "eta_source_adaptive", "eta_source_sequential")
    return levels


def _primary_action(rows):
    actions = {str(row.get("action")) for row in rows}
    if "joint_voi" in actions:
        return "joint_voi"
    if "hvd_voi" in actions:
        return "hvd_voi"
    return "new_only"


def _primary_hvd(rows):
    return (
        "factor_hierarchical"
        if any(str(row.get("hvd")) == "factor_hierarchical" for row in rows)
        else "factor_cumulative"
    )


def _primary_certificate(rows):
    return (
        "joint_tangent"
        if any(str(row.get("certificate")) == "joint_tangent" for row in rows)
        else "separable"
    )


def _primary_mean_profile(rows):
    profiles = {str(row.get("mean_profile")) for row in rows}
    if "eta_source_sequential" in profiles:
        return "eta_source_sequential"
    if "eta_source_adaptive" in profiles:
        return "eta_source_adaptive"
    return "legacy"


def _finite(values):
    rows = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            rows.append(value)
    return rows


def _median(values):
    values = _finite(values)
    return None if not values else float(statistics.median(values))


def _decision_diagnostic(row, key):
    value = row.get(key)
    if value is not None:
        return value
    diagnostics = row.get("decision_backend_diagnostics") or {}
    return diagnostics.get(key)


def _parse_variant(value):
    parts = str(value).strip("/").split("/")
    if not parts or parts[0] != "hvd_decision_gate":
        raise ValueError(f"unrecognized gate variant {value!r}")
    payload = parts[1:]
    shock = next(
        (part for part in reversed(payload) if part.startswith("shock")),
        None,
    )
    hvd = next((part for part in payload if part in {
        "pooled", "factor_cumulative", "factor_hierarchical",
    }), None)
    discrepancy = next((part for part in payload if part in {
        "frozen", "adaptive",
    }), None)
    action = next((part for part in payload if part in {
        "new_only", "hvd_voi", "joint_voi",
        "certificate_depth_new", "certificate_depth_search",
    }), None)
    certificate = next((part for part in payload if part in {
        "separable", "joint_tangent",
    }), "separable")
    mean_profile = next((part[len("mean_"):] for part in payload if part in {
        "mean_eta_empirical",
        "mean_eta_source_prior",
        "mean_eta_source_adaptive",
        "mean_eta_source_sequential",
    }), "legacy")
    if shock is None:
        raise ValueError(f"missing shock label in {value!r}")
    if hvd is None or discrepancy is None or action is None:
        raise ValueError(f"missing registered gate factor in {value!r}")
    return {
        "hvd": hvd,
        "discrepancy": discrepancy,
        "action": action,
        "certificate": certificate,
        "mean_profile": mean_profile,
        "shock_label": shock,
    }


def load_rows(root):
    rows = []
    errors = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = dict(payload["rows"][0])
            variant = _parse_variant(
                row.get("experiment_variant")
                or payload.get("experiment_variant"))
            row.update(variant)
            row["result_path"] = str(path)
            rows.append(row)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return rows, errors


def summarize_cell(items):
    feasible_regret = [
        row.get("feasible_simple_regret")
        for row in items
        if bool(row.get("true_feasible", False))
    ]
    nonvacuous = [
        not bool(row.get("posterior_certificate_vacuous", True))
        for row in items
    ]
    return {
        "count": int(len(items)),
        "true_feasible_count": int(sum(
            bool(row.get("true_feasible", False)) for row in items)),
        "posterior_feasible_count": int(sum(
            bool(row.get("posterior_feasible", False)) for row in items)),
        "false_feasible_count": int(sum(
            bool(row.get("false_feasible", False)) for row in items)),
        "false_certificate_count": int(sum(
            int(row.get("false_certificate_count") or 0) for row in items)),
        "certificate_nonvacuous_count": int(sum(nonvacuous)),
        "median_certificate_precision": _median(
            row.get("certificate_precision") for row in items),
        "median_feasible_regret": _median(feasible_regret),
        "median_initial_best_feasible_regret": _median(
            row.get("initial_best_feasible_regret") for row in items),
        "adaptive_rescue_count": int(sum(
            bool(row.get("adaptive_rescue", False)) for row in items)),
        "adaptive_loss_count": int(sum(
            bool(row.get("adaptive_loss", False)) for row in items)),
        "adaptive_improvement_count": int(sum(
            bool(row.get("adaptive_improves_initial_best", False))
            for row in items)),
        "median_adaptive_regret_change": _median(
            row.get("adaptive_regret_change") for row in items),
        "median_variance_log_rmse": _median(
            row.get("variance_log_rmse") for row in items),
        "median_certified_variance_log_rmse": _median(
            row.get("certified_variance_log_rmse") for row in items),
        "median_variance_upper_coverage": _median(
            row.get("variance_upper_coverage") for row in items),
        "median_predicted_true_variance_ratio": _median(
            row.get("median_predicted_true_variance_ratio") for row in items),
        "median_certified_true_variance_ratio": _median(
            row.get("median_certified_true_variance_ratio") for row in items),
        "selected_replication_count": int(sum(
            int(row.get("adaptive_replication_selected_count") or 0)
            for row in items)),
        "selected_new_point_count": int(sum(
            int(row.get("adaptive_new_point_selected_count") or 0)
            for row in items)),
        "median_selected_constraint_epistemic_information_reduction": _median(
            _decision_diagnostic(
                row,
                "mean_selected_constraint_epistemic_information_reduction",
            )
            for row in items),
        "median_selected_hvd_margin_information_reduction": _median(
            _decision_diagnostic(
                row,
                "mean_selected_hvd_margin_information_reduction",
            )
            for row in items),
        "median_selected_joint_information_reduction": _median(
            _decision_diagnostic(
                row,
                "mean_selected_joint_information_reduction",
            )
            for row in items),
    }


def cell_summaries(rows):
    grouped = {}
    for row in rows:
        key = (
            row["heldout"], row["shock_label"], row["hvd"],
            row["discrepancy"], row["action"], row["certificate"],
            row["mean_profile"],
        )
        grouped.setdefault(key, []).append(row)
    return {
        "/".join(key): summarize_cell(items)
        for key, items in sorted(grouped.items())
    }


def primary_cell_summaries(
    rows,
    primary_action=None,
    primary_hvd=None,
    primary_certificate=None,
    primary_mean_profile=None,
):
    primary_action = primary_action or _primary_action(rows)
    primary_hvd = primary_hvd or _primary_hvd(rows)
    primary_certificate = primary_certificate or _primary_certificate(rows)
    primary_mean_profile = primary_mean_profile or _primary_mean_profile(rows)
    primary = [
        row for row in rows
        if row["hvd"] == primary_hvd
        and row["discrepancy"] == "adaptive"
        and row["action"] == primary_action
        and row["certificate"] == primary_certificate
        and row["mean_profile"] == primary_mean_profile
    ]
    grouped = {}
    for row in primary:
        key = (row["heldout"], row["shock_label"])
        grouped.setdefault(key, []).append(row)
    return {
        "/".join(key): summarize_cell(items)
        for key, items in sorted(grouped.items())
    }


def paired_factor_effect(rows, factor, factors=None):
    factors = FACTORS if factors is None else factors
    control, challenger = factors[factor]
    other = [name for name in factors if name != factor]
    index = {}
    for row in rows:
        key = (
            row["heldout"], row["shock_label"], int(row["seed"]),
            *(row[name] for name in other),
        )
        index.setdefault(key, {})[row[factor]] = row
    pairs = [value for value in index.values() if control in value and challenger in value]
    feasible_wins = feasible_losses = regret_wins = regret_losses = 0
    rmse_delta = []
    coverage_delta = []
    nonvacuous_delta = []
    for pair in pairs:
        first, second = pair[control], pair[challenger]
        first_feasible = bool(first.get("true_feasible", False))
        second_feasible = bool(second.get("true_feasible", False))
        feasible_wins += int(second_feasible and not first_feasible)
        feasible_losses += int(first_feasible and not second_feasible)
        if first_feasible and second_feasible:
            first_regret = first.get("feasible_simple_regret")
            second_regret = second.get("feasible_simple_regret")
            if first_regret is not None and second_regret is not None:
                regret_wins += int(float(second_regret) < float(first_regret) - 1e-12)
                regret_losses += int(float(first_regret) < float(second_regret) - 1e-12)
        if first.get("variance_log_rmse") is not None and second.get(
            "variance_log_rmse") is not None:
            rmse_delta.append(
                float(second["variance_log_rmse"])
                - float(first["variance_log_rmse"]))
        if first.get("variance_upper_coverage") is not None and second.get(
            "variance_upper_coverage") is not None:
            coverage_delta.append(
                float(second["variance_upper_coverage"])
                - float(first["variance_upper_coverage"]))
        nonvacuous_delta.append(
            int(not bool(second.get("posterior_certificate_vacuous", True)))
            - int(not bool(first.get("posterior_certificate_vacuous", True)))
        )
    return {
        "control": control,
        "challenger": challenger,
        "pair_count": int(len(pairs)),
        "feasibility_wins": int(feasible_wins),
        "feasibility_losses": int(feasible_losses),
        "conditional_regret_wins": int(regret_wins),
        "conditional_regret_losses": int(regret_losses),
        "median_variance_log_rmse_delta": _median(rmse_delta),
        "median_variance_upper_coverage_delta": _median(coverage_delta),
        "certificate_nonvacuous_net": int(sum(nonvacuous_delta)),
    }


def paired_effects_by_stratum(rows, factors=None):
    factors = FACTORS if factors is None else factors
    grouped = {}
    for row in rows:
        key = (row["heldout"], row["shock_label"])
        grouped.setdefault(key, []).append(row)
    return {
        "/".join(key): {
            factor: paired_factor_effect(items, factor, factors)
            for factor in factors
        }
        for key, items in sorted(grouped.items())
    }


def load_identifiability_rows(root):
    rows = []
    errors = []
    if root is None:
        return rows, errors
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("experiment") != "hvd_identifiability":
                raise ValueError("not an HVD identifiability result")
            row = dict(payload)
            row["result_path"] = str(path)
            rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return rows, errors


def summarize_identifiability_cell(items):
    return {
        "count": int(len(items)),
        "median_log_variance_rmse": _median(
            row.get("log_variance_rmse") for row in items),
        "median_variance_spearman": _median(
            row.get("variance_spearman") for row in items),
        "median_shared_risk_spearman": _median(
            row.get("shared_risk_spearman") for row in items),
        "median_predicted_true_ratio": _median(
            row.get("median_predicted_true_ratio") for row in items),
        "median_certified_true_ratio": _median(
            row.get("median_certified_true_ratio") for row in items),
        "median_variance_upper_coverage": _median(
            row.get("variance_upper_coverage") for row in items),
        "certificate_nonvacuous_count": int(sum(
            bool(row.get("certificate_nonvacuous", False)) for row in items)),
        "false_feasible_count": int(sum(
            int(row.get("false_feasible_count") or 0) for row in items)),
        "median_certificate_precision": _median(
            row.get("certificate_precision") for row in items),
        "median_certificate_recall": _median(
            row.get("certificate_recall") for row in items),
    }


def identifiability_cell_summaries(rows):
    grouped = {}
    for row in rows:
        key = (
            str(row["mode"]),
            f"shock{float(row['shared_shock_scale']):g}".replace(".", "p"),
            f"rep{int(row['replicates_per_policy'])}",
        )
        grouped.setdefault(key, []).append(row)
    return {
        "/".join(key): summarize_identifiability_cell(items)
        for key, items in sorted(grouped.items())
    }


def paired_identifiability_effect(rows):
    index = {}
    for row in rows:
        key = (
            float(row["shared_shock_scale"]),
            int(row["replicates_per_policy"]),
            int(row["seed"]),
        )
        index.setdefault(key, {})[str(row["mode"])] = row
    pairs = [
        value for value in index.values()
        if "pooled" in value and "factor_cumulative" in value
    ]
    rmse_delta = []
    coverage_delta = []
    rank_delta = []
    nonvacuous_delta = []
    false_feasible_delta = []
    for pair in pairs:
        pooled = pair["pooled"]
        factor = pair["factor_cumulative"]
        if pooled.get("log_variance_rmse") is not None and factor.get(
            "log_variance_rmse") is not None:
            rmse_delta.append(
                float(factor["log_variance_rmse"])
                - float(pooled["log_variance_rmse"]))
        if pooled.get("variance_upper_coverage") is not None and factor.get(
            "variance_upper_coverage") is not None:
            coverage_delta.append(
                float(factor["variance_upper_coverage"])
                - float(pooled["variance_upper_coverage"]))
        if pooled.get("variance_spearman") is not None and factor.get(
            "variance_spearman") is not None:
            rank_delta.append(
                float(factor["variance_spearman"])
                - float(pooled["variance_spearman"]))
        nonvacuous_delta.append(
            int(bool(factor.get("certificate_nonvacuous", False)))
            - int(bool(pooled.get("certificate_nonvacuous", False))))
        false_feasible_delta.append(
            int(factor.get("false_feasible_count") or 0)
            - int(pooled.get("false_feasible_count") or 0))
    return {
        "pair_count": int(len(pairs)),
        "median_log_variance_rmse_delta": _median(rmse_delta),
        "median_variance_upper_coverage_delta": _median(coverage_delta),
        "median_variance_spearman_delta": _median(rank_delta),
        "certificate_nonvacuous_net": int(sum(nonvacuous_delta)),
        "false_feasible_net": int(sum(false_feasible_delta)),
    }


def identifiability_replication_trends(rows):
    grouped = {}
    for row in rows:
        key = (str(row["mode"]), float(row["shared_shock_scale"]), int(row["seed"]))
        grouped.setdefault(key, {})[int(row["replicates_per_policy"])] = row
    changes = {}
    for (mode, shock, _), by_replication in grouped.items():
        if 2 not in by_replication or 16 not in by_replication:
            continue
        key = (mode, f"shock{shock:g}".replace(".", "p"))
        first = by_replication[2]
        last = by_replication[16]
        record = changes.setdefault(key, {
            "rmse": [], "coverage": [], "rank": [], "shared_rank": []})
        record["rmse"].append(
            float(last["log_variance_rmse"])
            - float(first["log_variance_rmse"]))
        record["coverage"].append(
            float(last["variance_upper_coverage"])
            - float(first["variance_upper_coverage"]))
        if first.get("variance_spearman") is not None and last.get(
            "variance_spearman") is not None:
            record["rank"].append(
                float(last["variance_spearman"])
                - float(first["variance_spearman"]))
        if first.get("shared_risk_spearman") is not None and last.get(
            "shared_risk_spearman") is not None:
            record["shared_rank"].append(
                float(last["shared_risk_spearman"])
                - float(first["shared_risk_spearman"]))
    return {
        "/".join(key): {
            "pair_count": int(len(value["rmse"])),
            "rep16_minus_rep2_log_variance_rmse": _median(value["rmse"]),
            "rep16_minus_rep2_variance_upper_coverage": _median(value["coverage"]),
            "rep16_minus_rep2_variance_spearman": _median(value["rank"]),
            "rep16_minus_rep2_shared_risk_spearman": _median(
                value["shared_rank"]),
        }
        for key, value in sorted(changes.items())
    }


def summarize_identifiability(rows, errors, expected_count):
    paired = paired_identifiability_effect(rows)
    return {
        "expected_count": int(expected_count),
        "parsed_count": int(len(rows)),
        "errors": errors,
        "complete_expected_matrix": bool(
            not errors and len(rows) == int(expected_count)),
        "cells": identifiability_cell_summaries(rows),
        "paired_factor_vs_pooled": paired,
        "replication_trends": identifiability_replication_trends(rows),
    }


def gate_decision(
    rows,
    errors,
    expected_count,
    primary_action=None,
    primary_hvd=None,
    primary_certificate=None,
    primary_mean_profile=None,
):
    primary_action = primary_action or _primary_action(rows)
    primary_hvd = primary_hvd or _primary_hvd(rows)
    primary_certificate = primary_certificate or _primary_certificate(rows)
    primary_mean_profile = primary_mean_profile or _primary_mean_profile(rows)
    primary = [
        row for row in rows
        if row["hvd"] == primary_hvd
        and row["discrepancy"] == "adaptive"
        and row["action"] == primary_action
        and row["certificate"] == primary_certificate
        and row["mean_profile"] == primary_mean_profile
    ]
    false_certificates = sum(
        int(row.get("false_certificate_count") or 0) for row in primary)
    nonvacuous = sum(
        not bool(row.get("posterior_certificate_vacuous", True))
        for row in primary)
    online_gain = sum(
        bool(row.get("adaptive_rescue", False))
        or bool(row.get("adaptive_improves_initial_best", False))
        for row in primary
    )
    adaptive_losses = sum(
        bool(row.get("adaptive_loss", False)) for row in primary)
    coverage = _median(row.get("variance_upper_coverage") for row in primary)
    complete = not errors and len(rows) == int(expected_count)
    criteria = {
        "complete_expected_matrix": bool(complete),
        "zero_false_certificates": bool(false_certificates == 0),
        "nonvacuous_certificate_exists": bool(nonvacuous > 0),
        "median_variance_upper_coverage_at_least_0p9": bool(
            coverage is not None and coverage >= 0.90),
        "online_gain_exceeds_adaptive_loss": bool(online_gain > adaptive_losses),
    }
    return {
        "advance_to_20_seeds": bool(all(criteria.values())),
        "criteria": criteria,
        "primary_count": int(len(primary)),
        "primary_nonvacuous_count": int(nonvacuous),
        "primary_false_certificate_count": int(false_certificates),
        "primary_online_gain_count": int(online_gain),
        "primary_adaptive_loss_count": int(adaptive_losses),
        "primary_median_variance_upper_coverage": coverage,
        "primary_action": str(primary_action),
        "primary_hvd": str(primary_hvd),
        "primary_certificate": str(primary_certificate),
        "primary_mean_profile": str(primary_mean_profile),
    }


def analyze(
    root,
    expected_count=240,
    identifiability_root=None,
    expected_identifiability_count=160,
):
    rows, errors = load_rows(root)
    factors = _factor_levels(rows)
    primary_action = _primary_action(rows)
    primary_hvd = _primary_hvd(rows)
    primary_certificate = _primary_certificate(rows)
    primary_mean_profile = _primary_mean_profile(rows)
    ident_rows, ident_errors = load_identifiability_rows(identifiability_root)
    identifiability = (
        summarize_identifiability(
            ident_rows,
            ident_errors,
            expected_identifiability_count,
        )
        if identifiability_root is not None
        else None
    )
    gate = gate_decision(
        rows,
        errors,
        expected_count,
        primary_action=primary_action,
        primary_hvd=primary_hvd,
        primary_certificate=primary_certificate,
        primary_mean_profile=primary_mean_profile,
    )
    if identifiability is not None:
        ident_effect = identifiability["paired_factor_vs_pooled"]
        ident_criteria = {
            "complete_controlled_identifiability_matrix": bool(
                identifiability["complete_expected_matrix"]),
            "factor_cumulative_improves_controlled_log_rmse": bool(
                ident_effect["median_log_variance_rmse_delta"] is not None
                and ident_effect["median_log_variance_rmse_delta"] < 0.0),
        }
        gate["criteria"].update(ident_criteria)
        gate["advance_to_20_seeds"] = bool(all(gate["criteria"].values()))
    return {
        "schema_version": 2,
        "result_root": str(Path(root)),
        "expected_count": int(expected_count),
        "parsed_count": int(len(rows)),
        "errors": errors,
        "factor_levels": factors,
        "primary_action": primary_action,
        "primary_hvd": primary_hvd,
        "primary_certificate": primary_certificate,
        "primary_mean_profile": primary_mean_profile,
        "cells": cell_summaries(rows),
        "primary_cells": primary_cell_summaries(
            rows,
            primary_action=primary_action,
            primary_hvd=primary_hvd,
            primary_certificate=primary_certificate,
            primary_mean_profile=primary_mean_profile,
        ),
        "paired_effects": {
            factor: paired_factor_effect(rows, factor, factors)
            for factor in factors
        },
        "paired_effects_by_stratum": paired_effects_by_stratum(
            rows, factors),
        "identifiability": identifiability,
        "gate": gate,
    }


def markdown_report(result):
    gate = result["gate"]
    lines = [
        "# HVD decision closure gate",
        "",
        f"- Parsed: {result['parsed_count']}/{result['expected_count']}",
        f"- Advance to 20 seeds: `{gate['advance_to_20_seeds']}`",
        f"- Primary HVD: `{gate['primary_hvd']}`",
        f"- Primary action: `{gate['primary_action']}`",
        f"- Primary certificate: `{gate['primary_certificate']}`",
        f"- Primary mean profile: `{gate['primary_mean_profile']}`",
        f"- Primary nonvacuous certificates: {gate['primary_nonvacuous_count']}",
        f"- Primary false certificates: {gate['primary_false_certificate_count']}",
        f"- Primary online gains / losses: {gate['primary_online_gain_count']} / "
        f"{gate['primary_adaptive_loss_count']}",
        "",
        "## Paired effects",
        "",
        "| factor | pairs | feasible +/- | regret +/- | median log-RMSE delta | coverage delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for factor, effect in result["paired_effects"].items():
        lines.append(
            f"| {factor} | {effect['pair_count']} | "
            f"{effect['feasibility_wins']}/{effect['feasibility_losses']} | "
            f"{effect['conditional_regret_wins']}/{effect['conditional_regret_losses']} | "
            f"{effect['median_variance_log_rmse_delta']} | "
            f"{effect['median_variance_upper_coverage_delta']} |"
        )
    lines.extend([
        "",
        "## Primary cells",
        "",
        "| stratum | feasible | regret | n0 regret | rescue/loss | reps/new | GPR/HVD info | log-RMSE | ratio | coverage | certs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for stratum, cell in result["primary_cells"].items():
        lines.append(
            f"| {stratum} | {cell['true_feasible_count']}/{cell['count']} | "
            f"{cell['median_feasible_regret']} | "
            f"{cell['median_initial_best_feasible_regret']} | "
            f"{cell['adaptive_rescue_count']}/{cell['adaptive_loss_count']} | "
            f"{cell['selected_replication_count']}/{cell['selected_new_point_count']} | "
            f"{cell['median_selected_constraint_epistemic_information_reduction']}/"
            f"{cell['median_selected_hvd_margin_information_reduction']} | "
            f"{cell['median_variance_log_rmse']} | "
            f"{cell['median_predicted_true_variance_ratio']} | "
            f"{cell['median_variance_upper_coverage']} | "
            f"{cell['certificate_nonvacuous_count']} |"
        )
    if result.get("identifiability") is not None:
        ident = result["identifiability"]
        effect = ident["paired_factor_vs_pooled"]
        lines.extend([
            "",
            "## Controlled HVD identifiability",
            "",
            f"- Parsed: {ident['parsed_count']}/{ident['expected_count']}",
            f"- Factor minus pooled median log-RMSE: "
            f"{effect['median_log_variance_rmse_delta']}",
            f"- Factor minus pooled median coverage: "
            f"{effect['median_variance_upper_coverage_delta']}",
            f"- Factor minus pooled median rank correlation: "
            f"{effect['median_variance_spearman_delta']}",
            "",
            "| mode/shock | pairs | rep16-rep2 RMSE | coverage | rank | shared rank |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for key, trend in ident["replication_trends"].items():
            lines.append(
                f"| {key} | {trend['pair_count']} | "
                f"{trend['rep16_minus_rep2_log_variance_rmse']} | "
                f"{trend['rep16_minus_rep2_variance_upper_coverage']} | "
                f"{trend['rep16_minus_rep2_variance_spearman']} | "
                f"{trend['rep16_minus_rep2_shared_risk_spearman']} |"
            )
    lines.extend(["", "## Promotion criteria", ""])
    for name, passed in gate["criteria"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=240)
    parser.add_argument("--identifiability-root", type=Path)
    parser.add_argument("--expected-identifiability-count", type=int, default=160)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.root,
        expected_count=args.expected_count,
        identifiability_root=args.identifiability_root,
        expected_identifiability_count=args.expected_identifiability_count,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({
        "parsed_count": result["parsed_count"],
        "advance_to_20_seeds": result["gate"]["advance_to_20_seeds"],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }))


if __name__ == "__main__":
    main()
