#!/usr/bin/env python3
"""Decompose empty theory certificates into their conservative layers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.benchmark_quality import parse_weights


def _median(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(np.median(finite))


def _stage(row, name):
    audit = row.get("certification_margin_decomposition") or {}
    minimum = audit.get("minimum_margin") or {}
    return minimum.get(name) or {}


def _source_mean_diagnostics(row):
    for model in row.get("gpr_numerics") or []:
        diagnostics = model.get("source_parametric_prior") or {}
        if diagnostics:
            return diagnostics
    return {}


def _hvd_target_null_mass(row):
    diagnostics = row.get("variance_diagnostics") or {}
    domains = (diagnostics.get("cumulative_prior_component_domains") or {}).get(
        "1", [])
    weights = (diagnostics.get("cumulative_prior_component_weights") or {}).get(
        "1")
    if weights is None or "target:null" not in domains:
        return None
    weights = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if len(weights) != len(domains) or float(np.sum(weights)) <= 0.0:
        return None
    return float(weights[domains.index("target:null")] / np.sum(weights))


def _constraint_variance_diagnostic(row, name, default=None):
    diagnostics = row.get("variance_diagnostics") or {}
    value = diagnostics.get(name, default)
    if isinstance(value, dict):
        return value.get("1", default)
    return value


def _cell_key(path, row):
    adaptation_mode = str(row.get(
        "source_constraint_mean_adaptation_mode", "frozen"
    )).lower()
    mean_profile = (
        "legacy"
        if not bool(row.get("meta_observable_mean_coordinate", False))
        else (
            (
                "eta_source_sequential"
                if adaptation_mode in {
                    "sequential_evidence_mixture", "sequential_mixture"
                }
                else (
                    "eta_source_adaptive"
                    if adaptation_mode in {
                        "evidence_mixture", "mixture", "adaptive_mixture"
                    }
                    else "eta_source_prior"
                )
            )
            if bool(row.get(
                "source_constraint_mean_coefficient_prior", False))
            else "eta_empirical"
        )
    )
    variant = str(row.get("experiment_variant", ""))
    action = str(row.get("decision_backend", "unknown"))
    for registered in (
        "certificate_depth_search",
        "certificate_depth_new",
        "joint_voi",
        "hvd_voi",
        "new_only",
    ):
        if f"/{registered}/" in f"/{variant}/":
            action = registered
            break
    return (
        str(row.get("heldout", path.parent.parent.name)),
        str(row.get("hvd_ablation_profile", "unknown")),
        mean_profile,
        str(row.get("meta_observable_mean_mode", "unknown")),
        "adaptive" if row.get("source_discrepancy_update", False) else "frozen",
        action,
        float(row.get("target_shared_shock_scale", 1.0)),
        str(row.get(
            "task_posterior_robust_certificate_mode", "separable")),
        str(row.get("hvd_source_task_weight_mode", "independent")),
        str(row.get(
            "hvd_cumulative_target_evidence_mode",
            _constraint_variance_diagnostic(
                row, "cumulative_target_evidence_mode", "replication_only"),
        )),
    )


def oracle_variance_certifiability(row, config=None):
    """Replace only HVD variance by target truth in a post-run audit.

    Target truth is reconstructed after the decision and is never written back
    into a checkpoint or model.  The second margin also removes GPR epistemic
    uncertainty, separating a variance-transfer failure from a mean/posterior
    failure.
    """

    config = dict(config or {})
    x = row.get("recommendation_best_true_feasible_x")
    required = (
        row.get("recommendation_best_true_feasible_mu_con"),
        row.get("recommendation_best_true_feasible_epistemic_var"),
    )
    if x is None or any(value is None for value in required):
        return {"status": "unavailable"}
    heldout = str(row.get("heldout", ""))
    if not heldout:
        return {"status": "unavailable"}
    d = int(config.get("d", len(x)))
    if len(x) != d:
        return {"status": "dimension_mismatch"}
    problem_kwargs = {}
    if heldout == "FactorShockStatePolicyRZDT1":
        problem_kwargs["shared_shock_scale"] = float(row.get(
            "target_shared_shock_scale",
            config.get("target_shared_shock_scale", 1.0),
        ))
    problem = build_scalarized_problem(
        heldout,
        d,
        int(config.get("L", 100)),
        float(config.get("sigma", 0.04)),
        float(config.get("alpha", 0.05)),
        parse_weights(config.get("weights", "0.5,0.5")),
        problem_kwargs=problem_kwargs,
    )
    point = tuple(int(value) for value in x)
    true_sigma = float(problem.true_sigma(point)[1])
    true_mean = float(problem.true_constraint_mean(point))
    mu = float(row["recommendation_best_true_feasible_mu_con"])
    epistemic_var = max(float(
        row["recommendation_best_true_feasible_epistemic_var"]), 0.0)
    beta_g = max(float(config.get("beta_g", 2.0)), 0.0)
    z_alpha = float(norm.ppf(1.0 - float(config.get(
        "alpha", getattr(problem, "alpha", 0.05)))))
    mean_aleatoric_margin = (
        mu + z_alpha * true_sigma - float(problem.tau))
    oracle_variance_margin = (
        mean_aleatoric_margin + np.sqrt(beta_g * epistemic_var))
    true_chance_margin = true_mean + z_alpha * true_sigma - float(problem.tau)
    epistemic_radius = float(np.sqrt(beta_g * epistemic_var))
    oracle_mean_margin = (
        true_chance_margin + epistemic_radius)
    safety_depth = max(-float(true_chance_margin), 0.0)
    if safety_depth <= 0.0:
        epistemic_variance_contraction_factor = None
    elif epistemic_radius <= 0.0:
        epistemic_variance_contraction_factor = 0.0
    else:
        # Posterior variance, rather than its radius, must contract by this
        # factor before the same point is certifiable even with oracle mean.
        epistemic_variance_contraction_factor = float(
            (epistemic_radius / safety_depth) ** 2)
    return {
        "status": "audited",
        "oracle_variance_margin": float(oracle_variance_margin),
        "mean_aleatoric_margin": float(mean_aleatoric_margin),
        "oracle_variance_certified": bool(oracle_variance_margin <= 0.0),
        "mean_aleatoric_certified": bool(mean_aleatoric_margin <= 0.0),
        "true_constraint_sigma": true_sigma,
        "true_constraint_mean": true_mean,
        "posterior_mean_bias": float(mu - true_mean),
        "true_chance_margin": float(true_chance_margin),
        "oracle_mean_epistemic_radius": epistemic_radius,
        "epistemic_variance_contraction_factor": (
            epistemic_variance_contraction_factor),
        "oracle_mean_with_epistemic_margin": float(oracle_mean_margin),
        "target_oracle_used_for_decision": False,
        "target_oracle_used_for_post_run_audit": True,
    }


def summarize(paths):
    cells = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        config = dict(payload.get("config") or {})
        row = (
            dict(payload["rows"][0])
            if isinstance(payload.get("rows"), list) and payload["rows"]
            else payload
        )
        if row.get("certification_margin_decomposition"):
            row["oracle_variance_certifiability"] = (
                oracle_variance_certifiability(row, config))
            cells[_cell_key(path, row)].append(row)

    summaries = []
    for key, rows in sorted(cells.items()):
        (
            heldout,
            hvd,
            mean_profile,
            observable_mean_mode,
            discrepancy,
            backend,
            shock,
            certificate_mode,
            hvd_task_weight_mode,
            hvd_target_evidence_mode,
        ) = key
        nominal = [_stage(row, "observation_nominal") for row in rows]
        expert = [_stage(row, "expert_certified") for row in rows]
        robust = [_stage(row, "task_robust") for row in rows]
        separable = [_stage(row, "task_separable_robust") for row in rows]
        joint = [_stage(row, "task_joint_robust") for row in rows]
        final = [_stage(row, "final_certificate") for row in rows]
        nominal_margin = [stage.get("margin") for stage in nominal]
        expert_margin = [stage.get("margin") for stage in expert]
        robust_margin = [stage.get("margin") for stage in robust]
        final_margin = [stage.get("margin") for stage in final]
        oracle = [
            row.get("oracle_variance_certifiability") or {}
            for row in rows
        ]
        oracle_available = [
            audit for audit in oracle if audit.get("status") == "audited"
        ]
        source_mean = [_source_mean_diagnostics(row) for row in rows]
        adaptive_source_mean = [
            diagnostics for diagnostics in source_mean
            if diagnostics.get("adaptation_mode") in {
                "target_evidence_mixture",
                "sequential_target_evidence_mixture",
            }
        ]
        valid_layers = all(stage for stage in nominal + expert + robust)
        summary = {
            "heldout": heldout,
            "hvd": hvd,
            "mean_profile": mean_profile,
            "observable_mean_mode": observable_mean_mode,
            "discrepancy": discrepancy,
            "backend": backend,
            "shock_scale": shock,
            "certificate_mode": certificate_mode,
            "hvd_task_weight_mode": hvd_task_weight_mode,
            "hvd_target_evidence_mode": hvd_target_evidence_mode,
            "n": len(rows),
            "true_feasible_runs": int(sum(
                bool(row.get("true_feasible", False)) for row in rows)),
            "false_certificate_count": int(sum(
                int(row.get("false_certificate_count") or 0) for row in rows)),
            "adaptive_loss_count": int(sum(
                bool(row.get("adaptive_loss", False)) for row in rows)),
            "median_feasible_regret": _median([
                row.get("feasible_simple_regret") for row in rows]),
            "median_pool_min_true_margin": _median([
                (row.get("truth_pool_diagnostics") or {}).get(
                    "mean_pool_min_true_margin")
                for row in rows
            ]),
            "median_pool_min_posterior_margin": _median([
                (row.get("truth_pool_diagnostics") or {}).get(
                    "mean_pool_min_posterior_margin")
                for row in rows
            ]),
            "median_selected_true_margin": _median([
                (row.get("truth_pool_diagnostics") or {}).get(
                    "mean_selected_true_margin")
                for row in rows
            ]),
            "median_variance_log_rmse": _median([
                row.get("variance_log_rmse") for row in rows]),
            "median_predicted_true_variance_ratio": _median([
                row.get("median_predicted_true_variance_ratio") for row in rows]),
            "certified_runs": int(sum(
                int((row.get("certification_margin_decomposition") or {}).get(
                    "n_certified", 0)) > 0
                for row in rows
            )),
            "median_min_final_margin": _median(final_margin),
            "oracle_variance_audit_count": int(len(oracle_available)),
            "oracle_variance_certified_count": int(sum(
                bool(audit.get("oracle_variance_certified", False))
                for audit in oracle_available
            )),
            "mean_aleatoric_certified_count": int(sum(
                bool(audit.get("mean_aleatoric_certified", False))
                for audit in oracle_available
            )),
            "median_oracle_variance_margin": _median([
                audit.get("oracle_variance_margin")
                for audit in oracle_available
            ]),
            "median_mean_aleatoric_margin": _median([
                audit.get("mean_aleatoric_margin")
                for audit in oracle_available
            ]),
            "median_true_chance_margin": _median([
                audit.get("true_chance_margin")
                for audit in oracle_available
            ]),
            "median_posterior_mean_bias": _median([
                audit.get("posterior_mean_bias")
                for audit in oracle_available
            ]),
            "median_oracle_mean_with_epistemic_margin": _median([
                audit.get("oracle_mean_with_epistemic_margin")
                for audit in oracle_available
            ]),
            "median_epistemic_variance_contraction_factor": _median([
                audit.get("epistemic_variance_contraction_factor")
                for audit in oracle_available
            ]),
            "median_final_mean_term": _median([
                stage.get("mean_minus_tau") for stage in final]),
            "median_final_epistemic_radius": _median([
                stage.get("epistemic_radius") for stage in final]),
            "median_final_aleatoric_radius": _median([
                stage.get("aleatoric_radius") for stage in final]),
            "source_mean_mixture_count": int(len(adaptive_source_mean)),
            "median_target_only_posterior_weight": _median([
                diagnostics.get("target_only_posterior_weight")
                for diagnostics in adaptive_source_mean
            ]),
            "median_source_posterior_weight": _median([
                diagnostics.get("source_posterior_weight")
                for diagnostics in adaptive_source_mean
            ]),
            "target_only_selected_count": int(sum(
                diagnostics.get("selected_component") == "target:null"
                for diagnostics in adaptive_source_mean
            )),
            "median_hvd_target_null_mass": _median([
                _hvd_target_null_mass(row) for row in rows
            ]),
            "median_hvd_target_shape_dof": _median([
                _constraint_variance_diagnostic(
                    row, "cumulative_prior_shape_target_dof", 0.0)
                for row in rows
            ]),
            "median_prequential_upper_solution_count": _median([
                (
                    _constraint_variance_diagnostic(
                        row, "prequential_upper_solution_count", 0)
                    if hvd_target_evidence_mode == "prequential_upper"
                    else 0
                )
                for row in rows
            ]),
            "median_hvd_target_weight": _median([
                _constraint_variance_diagnostic(
                    row, "cumulative_prior_target_weight", 0)
                for row in rows
            ]),
            "hvd_scale_sources": sorted({
                str(source)
                for row in rows
                for source in [
                    _constraint_variance_diagnostic(
                        row, "cumulative_prior_scale_source")
                ]
                if source is not None
            }),
            "median_nominal_margin": _median(nominal_margin),
            "median_expert_certified_margin": _median(expert_margin),
            "median_task_robust_margin": _median(robust_margin),
            "median_task_separable_margin": _median([
                stage.get("margin") for stage in separable]),
            "median_task_joint_margin": _median([
                stage.get("margin") for stage in joint]),
            "median_joint_tightening": _median([
                ((row.get("certification_margin_decomposition") or {}).get(
                    "minimum_margin") or {}).get("task_joint_tightening")
                for row in rows
            ]),
            "nominal_safe_expert_unsafe": None,
            "expert_safe_robust_unsafe": None,
            "median_expert_observation_aleatoric_ratio": _median([
                ((row.get("certification_margin_decomposition") or {}).get(
                    "minimum_margin") or {}).get(
                        "expert_to_observation_aleatoric_ratio")
                for row in rows
            ]),
            "median_robust_expert_aleatoric_ratio": _median([
                ((row.get("certification_margin_decomposition") or {}).get(
                    "minimum_margin") or {}).get(
                        "robust_to_expert_aleatoric_ratio")
                for row in rows
            ]),
        }
        if valid_layers:
            summary["nominal_safe_expert_unsafe"] = int(sum(
                float(nominal_margin[index]) <= 0.0
                and float(expert_margin[index]) > 0.0
                for index in range(len(rows))
            ))
            summary["expert_safe_robust_unsafe"] = int(sum(
                float(expert_margin[index]) <= 0.0
                and float(robust_margin[index]) > 0.0
                for index in range(len(rows))
            ))
        summaries.append(summary)
    return summaries


def render_markdown(summaries):
    lines = [
        "# Certification vacuity decomposition",
        "",
        (
            "Margins are measured at each run's minimum-final-margin candidate. "
            "The three stages isolate nominal posterior uncertainty, per-expert "
            "HVD certification guards, and KL-robust task mixing."
        ),
        "",
        "| domain | shock | HVD | mean | mean coord | shape task | shape evidence | target dof | preq admitted | target weight | scale source | action | certificate | n | feasible | cert | oracle-var cert | no-epi cert | false cert | loss | regret | var RMSE | pred/true | mean null | HVD null | null selected | nominal | expert-cert | separable | joint | tightening | active robust | final | true margin | mean bias | oracle-mean+epi | epi-var shrink | oracle-var margin | no-epi margin | mean term | epi | alea |",
        "|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines[-2] = lines[-2].replace(
        "| regret | var RMSE |",
        "| regret | pool true min | pool posterior min | selected true | var RMSE |",
    )
    lines[-1] = "|" + "|".join(
        ["---"] * (lines[-2].count("|") - 1)) + "|"
    for row in summaries:
        def fmt(value):
            return "NA" if value is None else f"{float(value):.4g}"

        lines.append(
            "| {heldout} | {shock_scale:g} | {hvd} | {mean_profile} | "
            "{observable_mean_mode} | "
            "{hvd_task_weight_mode} | "
            "{hvd_target_evidence_mode} | {target_dof} | {preq_count} | "
            "{target_weight} | {scale_source} | "
            "{backend} | "
            "{certificate_mode} | {n} | "
            "{true_feasible_runs} | {certified_runs} | "
            "{oracle_variance_certified_count}/{oracle_variance_audit_count} | "
            "{mean_aleatoric_certified_count}/{oracle_variance_audit_count} | "
            "{false_certificate_count} | "
            "{adaptive_loss_count} | {regret} | {pool_true_min} | "
            "{pool_posterior_min} | {selected_true} | {rmse} | {pred_ratio} | "
            "{null_weight} | {hvd_null} | "
            "{target_only_selected_count}/{source_mean_mixture_count} | "
            "{nominal} | {expert} | {separable} | {joint} | {tightening} | "
            "{robust} | {final} | "
            "{true_margin} | {mean_bias} | {oracle_mean_margin} | "
            "{epistemic_shrink} | "
            "{oracle_margin} | {no_epi_margin} | "
            "{mean} | {epi} | {alea} |".format(
                **row,
                nominal=fmt(row["median_nominal_margin"]),
                regret=fmt(row["median_feasible_regret"]),
                pool_true_min=fmt(row["median_pool_min_true_margin"]),
                pool_posterior_min=fmt(
                    row["median_pool_min_posterior_margin"]),
                selected_true=fmt(row["median_selected_true_margin"]),
                rmse=fmt(row["median_variance_log_rmse"]),
                pred_ratio=fmt(row["median_predicted_true_variance_ratio"]),
                null_weight=fmt(
                    row["median_target_only_posterior_weight"]),
                hvd_null=fmt(row["median_hvd_target_null_mass"]),
                target_dof=fmt(row["median_hvd_target_shape_dof"]),
                preq_count=fmt(
                    row["median_prequential_upper_solution_count"]),
                target_weight=fmt(row["median_hvd_target_weight"]),
                scale_source=(
                    ",".join(row["hvd_scale_sources"])
                    if row["hvd_scale_sources"] else "NA"
                ),
                expert=fmt(row["median_expert_certified_margin"]),
                separable=fmt(row["median_task_separable_margin"]),
                joint=fmt(row["median_task_joint_margin"]),
                tightening=fmt(row["median_joint_tightening"]),
                robust=fmt(row["median_task_robust_margin"]),
                final=fmt(row["median_min_final_margin"]),
                oracle_margin=fmt(row["median_oracle_variance_margin"]),
                no_epi_margin=fmt(row["median_mean_aleatoric_margin"]),
                true_margin=fmt(row["median_true_chance_margin"]),
                mean_bias=fmt(row["median_posterior_mean_bias"]),
                oracle_mean_margin=fmt(
                    row["median_oracle_mean_with_epistemic_margin"]),
                epistemic_shrink=fmt(
                    row["median_epistemic_variance_contraction_factor"]),
                mean=fmt(row["median_final_mean_term"]),
                epi=fmt(row["median_final_epistemic_radius"]),
                alea=fmt(row["median_final_aleatoric_radius"]),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = sorted(
        path
        for root in args.root
        for path in root.rglob("result.json")
    )
    summaries = summarize(paths)
    markdown = render_markdown(summaries)
    if args.out is None:
        print(markdown, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
        args.out.with_suffix(".json").write_text(
            json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
