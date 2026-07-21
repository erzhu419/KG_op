#!/usr/bin/env python3
"""Aggregate SC-OLH and transfer result matrices without copying run state.

The reader intentionally accepts only files named ``result.json``. Runtime
checkpoints, pickle files, model weights, and arbitrary result-directory
contents are never opened or copied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROW_FIELDS = (
    "run_id",
    "track",
    "variant",
    "method",
    "implementation",
    "initial_design",
    "initial_design_fingerprint",
    "source_archive_fingerprint",
    "proposal_mode",
    "proposal_structural_prior_profile",
    "proposal_source_dimension",
    "proposal_target_dimension",
    "domain",
    "seed",
    "d",
    "N",
    "n0",
    "source_calls",
    "total_calls",
    "d_over_target_calls",
    "d_over_total_calls",
    "status",
    "true_feasible",
    "feasible_regret",
    "true_objective",
    "constraint_violation",
    "initial_has_true_feasible",
    "initial_true_feasible_count",
    "initial_best_feasible_regret",
    "adaptive_rescue",
    "adaptive_loss",
    "adaptive_improves_initial_best",
    "adaptive_regret_change",
    "posterior_feasible",
    "posterior_certificate_vacuous",
    "posterior_certified_count",
    "false_certificate_count",
    "certificate_precision",
    "certificate_recall",
    "certified_true_feasible_count",
    "evaluated_point_count",
    "minimum_posterior_margin",
    "minimum_true_margin",
    "decision_backend",
    "terminal_value_contract",
    "decision_contract_coherent",
    "terminal_recommendation_observed_only",
    "audit_admissible_mainline",
    "source_oracle_aided",
    "target_oracle_used_for_adaptation",
    "target_oracle_used_for_decision",
    "online_updates_use_budgeted_target_only",
    "structural_prior_profile",
    "hvd_profile",
    "source_discrepancy_update",
    "recheck_top_k",
    "risk_penalty",
    "utility_weight",
    "adaptive_replication_voi",
    "adaptive_replication_count",
    "adaptive_new_point_count",
    "exact_mc_samples",
    "exact_sampling_mode",
    "exact_shortlist_size",
    "posterior_dominance_enabled",
    "posterior_dominance_switch_count",
    "wall_time_sec",
    "shared_shock_scale",
    "certification_tau",
    "replicates_per_policy",
    "log_variance_rmse",
    "variance_spearman",
    "shared_risk_spearman",
    "variance_upper_coverage",
    "true_feasible_rate",
    "posterior_feasible_rate_grid",
    "false_feasible_rate",
    "false_feasible_fraction_of_certified",
    "missed_feasible_rate",
    "missed_feasible_fraction_of_true",
    "median_predicted_true_ratio",
    "median_certified_true_ratio",
    "candidate_gen_time_total",
    "candidate_gen_time_share",
    "kg_compute_time_total",
    "kg_compute_time_share",
    "posterior_solve_time_total",
    "posterior_solve_time_share",
    "update_time_total",
    "update_time_share",
    "simulate_time_total",
    "simulate_time_share",
    "result_path",
)


TRACE_FIELDS = (
    "run_id",
    "track",
    "variant",
    "method",
    "implementation",
    "initial_design",
    "domain",
    "seed",
    "d",
    "N",
    "n0",
    "target_call",
    "iteration",
    "action_kind",
    "candidate_source",
    "x_fingerprint",
    "selected_score",
    "posterior_bayes_risk",
    "posterior_theory_margin",
    "posterior_constraint_epistemic",
    "observed_objective",
    "observed_constraint",
    "true_objective_post_run",
    "true_chance_margin_post_run",
    "true_feasible_post_run",
    "feasible_regret_post_run",
    "incumbent_feasible_regret_post_run",
    "exact_kg_best_new_raw",
    "exact_kg_best_replication_raw",
    "exact_kg_new_minus_replication_raw",
    "truth_join_timing",
    "target_oracle_used_for_decision",
    "result_path",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite(value)
    return None if number is None else int(number)


def _boolean(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    num = _finite(numerator)
    den = _finite(denominator)
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _stage_value(stage_times: dict, stage: str, field: str) -> float | None:
    return _finite(_dict(stage_times.get(stage)).get(field))


def _variant_parts(experiment_variant: str, run_id: str) -> tuple[str, str]:
    parts = [part for part in str(experiment_variant or "").split("/") if part]
    if len(parts) >= 2 and parts[0].startswith("paper_main_v1"):
        # Paper runs append the shock scenario after the registered method:
        # paper_main_v1_sequential/<method>/shock<scale>.
        return parts[0], parts[1]
    if len(parts) >= 3 and parts[0] == "structural_backend":
        return parts[-2], parts[-1]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    if parts:
        return "sc_olh", parts[-1]
    return "sc_olh", run_id


def _normalize_sc_result(
    payload: dict,
    row: dict,
    path: Path,
    root: Path,
) -> dict:
    config = _dict(payload.get("config"))
    audit = _dict(row.get("audit"))
    adaptation = _dict(row.get("source_target_adaptation_contract"))
    meta_prior = _dict(row.get("meta_prior"))
    meta_training = _dict(meta_prior.get("training"))
    certificate = _dict(row.get("certificate_outcome_audit"))
    adaptive = _dict(row.get("adaptive_outcome_audit"))
    dominance = _dict(row.get("posterior_dominance"))
    decision_contract = _dict(row.get("decision_backend_contract"))
    stage_times = _dict(row.get("stage_times"))
    experiment_variant = str(_first(
        row.get("experiment_variant"),
        payload.get("experiment_variant"),
        config.get("experiment_variant"),
        "",
    ))
    track, method = _variant_parts(experiment_variant, root.name)
    target_calls = _integer(_first(
        row.get("n_simulations"),
        adaptation.get("target_calls_used_for_adaptation"),
        row.get("N"),
        config.get("N"),
    ))
    source_call_candidates = [
        _integer(audit.get("source_simulator_calls")),
        _integer(adaptation.get("source_simulator_calls")),
    ]
    source_calls = next(
        (value for value in source_call_candidates
         if value is not None and value > 0),
        next((value for value in source_call_candidates if value is not None), None),
    )
    if (
        (source_calls is None or source_calls <= 0)
        and meta_training.get("source_observation_mode") == "replicated"
    ):
        source_records = _integer(_first(
            meta_training.get("n_records"), meta_prior.get("n_records")))
        source_replicates = _integer(
            meta_training.get("source_observation_replicates"))
        if (
            source_records is not None
            and source_records > 0
            and source_replicates is not None
            and source_replicates > 0
        ):
            source_calls = source_records * source_replicates
    dimension = _integer(_first(config.get("d"), row.get("d")))
    initial_count = _integer(_first(
        row.get("initial_true_feasible_count"),
        adaptive.get("initial_true_feasible_count"),
    ))
    initial_has = _boolean(_first(
        row.get("initial_has_true_feasible"),
        adaptive.get("initial_has_true_feasible"),
        None if initial_count is None else initial_count > 0,
    ))
    true_feasible = _boolean(_first(
        row.get("true_feasible"),
        adaptive.get("final_true_feasible"),
    ))
    regret = _finite(_first(
        row.get("feasible_simple_regret"),
        row.get("simple_regret"),
        adaptive.get("final_feasible_regret"),
    )) if true_feasible else None
    initial_regret = _finite(_first(
        row.get("initial_best_feasible_regret"),
        adaptive.get("initial_best_feasible_regret"),
    ))
    total_calls = (
        source_calls + target_calls
        if source_calls is not None and target_calls is not None
        else None
    )
    return {
        "run_id": root.name,
        "track": track,
        "variant": experiment_variant,
        "method": method,
        "implementation": "sc_olh",
        "initial_design": _first(config.get("initial_design"), row.get("task_initial_design")),
        "initial_design_fingerprint": _first(
            adaptation.get("target_initial_design_fingerprint"),
            config.get("initial_design_fingerprint"),
        ),
        "source_archive_fingerprint": _first(
            adaptation.get("source_archive_fingerprint"),
            adaptation.get("target_initial_design_source_archive_fingerprint"),
            config.get("initial_design_source_archive_fingerprint"),
        ),
        "proposal_mode": row.get("proposal_mode"),
        "proposal_structural_prior_profile": row.get(
            "proposal_structural_prior_profile"),
        "proposal_source_dimension": _integer(row.get(
            "proposal_source_dimension")),
        "proposal_target_dimension": _integer(row.get(
            "proposal_target_dimension")),
        "domain": row.get("heldout"),
        "seed": _integer(row.get("seed")),
        "d": dimension,
        "N": target_calls,
        "n0": _integer(_first(row.get("n0"), config.get("n0"))),
        "source_calls": source_calls,
        "total_calls": total_calls,
        "d_over_target_calls": _safe_ratio(dimension, target_calls),
        "d_over_total_calls": _safe_ratio(dimension, total_calls),
        "status": "ok",
        "true_feasible": true_feasible,
        "feasible_regret": regret,
        "true_objective": _finite(row.get("true_objective")),
        "constraint_violation": _finite(row.get("constraint_violation")),
        "initial_has_true_feasible": initial_has,
        "initial_true_feasible_count": initial_count,
        "initial_best_feasible_regret": initial_regret,
        "adaptive_rescue": _boolean(_first(
            row.get("adaptive_rescue"),
            adaptive.get("adaptive_rescue"),
            None if initial_has is None or true_feasible is None else (
                not initial_has and true_feasible
            ),
        )),
        "adaptive_loss": _boolean(_first(
            row.get("adaptive_loss"),
            adaptive.get("adaptive_loss"),
            None if initial_has is None or true_feasible is None else (
                initial_has and not true_feasible
            ),
        )),
        "adaptive_improves_initial_best": _boolean(_first(
            row.get("adaptive_improves_initial_best"),
            adaptive.get("adaptive_improves_initial_best"),
            None if regret is None or initial_regret is None else regret < initial_regret,
        )),
        "adaptive_regret_change": _finite(_first(
            row.get("adaptive_regret_change"),
            adaptive.get("adaptive_regret_change"),
            None if regret is None or initial_regret is None else regret - initial_regret,
        )),
        "posterior_feasible": _boolean(row.get("posterior_feasible")),
        "posterior_certificate_vacuous": _boolean(_first(
            row.get("posterior_certificate_vacuous"),
            certificate.get("posterior_certificate_vacuous"),
        )),
        "posterior_certified_count": _integer(_first(
            row.get("posterior_certified_evaluated_count"),
            certificate.get("posterior_certified_count"),
        )),
        "false_certificate_count": _integer(_first(
            row.get("false_certificate_count"),
            certificate.get("false_certificate_count"),
        )),
        "certificate_precision": _finite(_first(
            row.get("certificate_precision"),
            certificate.get("certificate_precision"),
        )),
        "certificate_recall": _finite(_first(
            row.get("certificate_recall_on_evaluated_feasible"),
            certificate.get("certificate_recall_on_evaluated_feasible"),
        )),
        "certified_true_feasible_count": _integer(
            certificate.get("certified_true_feasible_count")),
        "evaluated_point_count": _integer(
            certificate.get("evaluated_point_count")),
        "minimum_posterior_margin": _finite(
            certificate.get("minimum_posterior_margin")),
        "minimum_true_margin": _finite(
            certificate.get("minimum_true_margin")),
        "decision_backend": row.get("decision_backend"),
        "terminal_value_contract": decision_contract.get(
            "terminal_value_contract"),
        "decision_contract_coherent": _boolean(
            decision_contract.get("coherent")),
        "terminal_recommendation_observed_only": _boolean(
            decision_contract.get("terminal_recommendation_observed_only")),
        "audit_admissible_mainline": _boolean(_first(
            row.get("audit_admissible_mainline"),
            audit.get("admissible_mainline"),
        )),
        "source_oracle_aided": _boolean(
            adaptation.get("source_oracle_aided")),
        "target_oracle_used_for_adaptation": _boolean(
            adaptation.get("target_oracle_used_for_adaptation")),
        "target_oracle_used_for_decision": _boolean(_first(
            decision_contract.get("target_oracle_used"),
            certificate.get("target_oracle_used_for_decision"),
        )),
        "online_updates_use_budgeted_target_only": _boolean(
            decision_contract.get(
                "online_updates_use_budgeted_target_observations_only")),
        "structural_prior_profile": row.get("structural_prior_profile"),
        "hvd_profile": row.get("hvd_ablation_profile"),
        "source_discrepancy_update": _boolean(row.get("source_discrepancy_update")),
        "recheck_top_k": _integer(row.get("certification_recheck_top_k")),
        "risk_penalty": _finite(row.get("decision_risk_penalty")),
        "utility_weight": _finite(row.get("decision_source_utility_weight")),
        "adaptive_replication_voi": _boolean(row.get("adaptive_replication_voi_enabled")),
        "adaptive_replication_count": _integer(row.get("adaptive_replication_selected_count")),
        "adaptive_new_point_count": _integer(row.get("adaptive_new_point_selected_count")),
        "exact_mc_samples": _integer(_first(
            config.get("exact_kg_mc_samples"), row.get("exact_kg_mc_samples"))),
        "exact_sampling_mode": _first(
            config.get("exact_kg_sampling_mode"),
            row.get("exact_kg_sampling_mode"),
        ),
        "exact_shortlist_size": _integer(config.get(
            "evaluate_or_replicate_new_action_count")),
        "posterior_dominance_enabled": _boolean(_first(
            row.get("posterior_dominance_enabled"), dominance.get("enabled"))),
        "posterior_dominance_switch_count": _integer(_first(
            row.get("posterior_dominance_switch_count"), dominance.get("switch_count"))),
        "wall_time_sec": _finite(_first(row.get("wall_time_sec"), row.get("algorithm_time_sec"))),
        "shared_shock_scale": None,
        "replicates_per_policy": None,
        "log_variance_rmse": None,
        "variance_spearman": None,
        "shared_risk_spearman": None,
        "variance_upper_coverage": None,
        "false_feasible_rate": None,
        "missed_feasible_rate": None,
        "candidate_gen_time_total": _stage_value(
            stage_times, "t_candidate_gen", "total"),
        "candidate_gen_time_share": _stage_value(
            stage_times, "t_candidate_gen", "share"),
        "kg_compute_time_total": _stage_value(
            stage_times, "t_kg_compute", "total"),
        "kg_compute_time_share": _stage_value(
            stage_times, "t_kg_compute", "share"),
        "posterior_solve_time_total": _stage_value(
            stage_times, "t_posterior_solve", "total"),
        "posterior_solve_time_share": _stage_value(
            stage_times, "t_posterior_solve", "share"),
        "update_time_total": _stage_value(
            stage_times, "t_update", "total"),
        "update_time_share": _stage_value(
            stage_times, "t_update", "share"),
        "simulate_time_total": _stage_value(
            stage_times, "t_simulate", "total"),
        "simulate_time_share": _stage_value(
            stage_times, "t_simulate", "share"),
        "_online_action_trace": row.get("online_action_trace"),
        "result_path": str(path),
    }


def _normalize_transfer_result(payload: dict, path: Path, root: Path) -> dict:
    result = _dict(payload.get("result"))
    contract = _dict(payload.get("comparison_contract"))
    target = _dict(result.get("target_information_contract"))
    source = _dict(result.get("source_information_contract"))
    initial = _dict(result.get("initial_truth_audit"))
    posterior = _dict(result.get("posterior"))
    target_calls = _integer(_first(
        target.get("target_calls"),
        contract.get("target_total_calls_N"),
        result.get("n_simulations"),
    ))
    source_calls = _integer(_first(
        source.get("source_simulator_calls"),
        contract.get("source_simulator_calls"),
    ))
    dimension = _integer(_first(
        target.get("dimension"), contract.get("target_dimension")))
    total_calls = _integer(_first(
        contract.get("total_source_plus_target_calls"),
        None if source_calls is None or target_calls is None else source_calls + target_calls,
    ))
    initial_count = _integer(initial.get("true_feasible_count"))
    initial_has = None if initial_count is None else initial_count > 0
    true_feasible = _boolean(result.get("true_feasible"))
    regret = _finite(result.get("feasible_regret")) if true_feasible else None
    initial_regret = _finite(initial.get("best_true_feasible_regret"))
    final_improves = _boolean(initial.get("final_improves_initial_best"))
    posterior_margin = _finite(posterior.get("chance_bound"))
    return {
        "run_id": root.name,
        "track": "transfer",
        "variant": f"{payload.get('implementation')}/{payload.get('method')}",
        "method": payload.get("method"),
        "implementation": payload.get("implementation"),
        "initial_design": _first(target.get("initial_design"), contract.get("target_initial_design")),
        "proposal_mode": None,
        "proposal_structural_prior_profile": None,
        "proposal_source_dimension": None,
        "proposal_target_dimension": dimension,
        "domain": _first(payload.get("heldout_target_domain"), result.get("heldout")),
        "seed": _integer(payload.get("seed")),
        "d": dimension,
        "N": target_calls,
        "n0": _integer(_first(target.get("n0"), contract.get("target_initial_calls_n0"))),
        "source_calls": source_calls,
        "total_calls": total_calls,
        "d_over_target_calls": _safe_ratio(dimension, target_calls),
        "d_over_total_calls": _safe_ratio(dimension, total_calls),
        "status": payload.get("status", "ok"),
        "true_feasible": true_feasible,
        "feasible_regret": regret,
        "true_objective": _finite(result.get("true_objective")),
        "constraint_violation": _finite(result.get("constraint_violation")),
        "initial_has_true_feasible": initial_has,
        "initial_true_feasible_count": initial_count,
        "initial_best_feasible_regret": initial_regret,
        "adaptive_rescue": (
            None if initial_has is None or true_feasible is None
            else not initial_has and true_feasible
        ),
        "adaptive_loss": (
            None if initial_has is None or true_feasible is None
            else initial_has and not true_feasible
        ),
        "adaptive_improves_initial_best": final_improves,
        "adaptive_regret_change": (
            None if regret is None or initial_regret is None else regret - initial_regret
        ),
        "posterior_feasible": None if posterior_margin is None else posterior_margin <= 0,
        "posterior_certificate_vacuous": None,
        "posterior_certified_count": None,
        "false_certificate_count": None,
        "certificate_precision": None,
        "certificate_recall": None,
        "decision_backend": payload.get("method"),
        "structural_prior_profile": None,
        "hvd_profile": "pointwise_log_variance",
        "source_discrepancy_update": None,
        "recheck_top_k": None,
        "risk_penalty": None,
        "utility_weight": None,
        "adaptive_replication_voi": None,
        "adaptive_replication_count": None,
        "adaptive_new_point_count": None,
        "posterior_dominance_enabled": None,
        "posterior_dominance_switch_count": None,
        "wall_time_sec": _finite(_first(payload.get("wall_time_sec"), result.get("wall_time_sec"))),
        "shared_shock_scale": None,
        "replicates_per_policy": None,
        "log_variance_rmse": None,
        "variance_spearman": None,
        "shared_risk_spearman": None,
        "variance_upper_coverage": None,
        "false_feasible_rate": None,
        "missed_feasible_rate": None,
        "result_path": str(path),
    }


def _normalize_hvd_identifiability(payload: dict, path: Path, root: Path) -> dict:
    return {
        "run_id": root.name,
        "track": "hvd_identifiability",
        "variant": str(payload.get("mode")),
        "method": str(payload.get("mode")),
        "implementation": "orthogonal_hvd",
        "initial_design": "replicated_policy_design",
        "proposal_mode": None,
        "proposal_structural_prior_profile": None,
        "proposal_source_dimension": None,
        "proposal_target_dimension": _integer(payload.get("d")),
        "domain": "FactorShockStatePolicyRZDT1",
        "seed": _integer(payload.get("seed")),
        "d": _integer(payload.get("d")),
        "N": _integer(payload.get("simulator_calls")),
        "n0": _integer(payload.get("n_train_policies")),
        "source_calls": 0,
        "total_calls": _integer(payload.get("simulator_calls")),
        "d_over_target_calls": _safe_ratio(
            payload.get("d"), payload.get("simulator_calls")),
        "d_over_total_calls": _safe_ratio(
            payload.get("d"), payload.get("simulator_calls")),
        "status": payload.get("status", "ok"),
        "true_feasible": None,
        "feasible_regret": None,
        "true_objective": None,
        "constraint_violation": None,
        "initial_has_true_feasible": None,
        "initial_true_feasible_count": None,
        "initial_best_feasible_regret": None,
        "adaptive_rescue": None,
        "adaptive_loss": None,
        "adaptive_improves_initial_best": None,
        "adaptive_regret_change": None,
        "posterior_feasible": _boolean(payload.get("certificate_nonvacuous")),
        "posterior_certificate_vacuous": (
            None if payload.get("certificate_nonvacuous") is None
            else not bool(payload["certificate_nonvacuous"])
        ),
        "posterior_certified_count": _integer(payload.get(
            "posterior_feasible_count")),
        "false_certificate_count": _integer(payload.get(
            "false_feasible_count")),
        "certificate_precision": _finite(payload.get(
            "certificate_precision")),
        "certificate_recall": _finite(payload.get("certificate_recall")),
        "decision_backend": None,
        "structural_prior_profile": None,
        "hvd_profile": payload.get("mode"),
        "source_discrepancy_update": None,
        "recheck_top_k": None,
        "risk_penalty": None,
        "utility_weight": None,
        "adaptive_replication_voi": None,
        "adaptive_replication_count": None,
        "adaptive_new_point_count": None,
        "posterior_dominance_enabled": None,
        "posterior_dominance_switch_count": None,
        "wall_time_sec": _finite(payload.get("wall_time_sec")),
        "shared_shock_scale": _finite(payload.get("shared_shock_scale")),
        "certification_tau": _finite(payload.get("certification_tau")),
        "replicates_per_policy": _integer(payload.get(
            "replicates_per_policy")),
        "log_variance_rmse": _finite(payload.get("log_variance_rmse")),
        "variance_spearman": _finite(payload.get("variance_spearman")),
        "shared_risk_spearman": _finite(payload.get("shared_risk_spearman")),
        "variance_upper_coverage": _finite(payload.get(
            "variance_upper_coverage")),
        "true_feasible_rate": _finite(payload.get("true_feasible_rate")),
        "posterior_feasible_rate_grid": _finite(payload.get(
            "posterior_feasible_rate")),
        "false_feasible_rate": _finite(payload.get("false_feasible_rate")),
        "false_feasible_fraction_of_certified": _finite(payload.get(
            "false_feasible_fraction_of_certified")),
        "missed_feasible_rate": _finite(payload.get("missed_feasible_rate")),
        "missed_feasible_fraction_of_true": _finite(payload.get(
            "missed_feasible_fraction_of_true")),
        "median_predicted_true_ratio": _finite(payload.get(
            "median_predicted_true_ratio")),
        "median_certified_true_ratio": _finite(payload.get(
            "median_certified_true_ratio")),
        "result_path": str(path),
    }


def _iter_result_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("result.json")):
        lowered = {part.lower() for part in path.parts}
        if "checkpoints" in lowered or "checkpoint" in lowered:
            continue
        if path.name != "result.json" or path.suffix.lower() != ".json":
            continue
        yield path


def load_rows(roots: Iterable[Path]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    errors: list[dict] = []
    for root in roots:
        root = root.resolve()
        for path in _iter_result_paths(root):
            try:
                payload = json.loads(path.read_text())
                if isinstance(payload.get("rows"), list):
                    rows.extend(
                        _normalize_sc_result(payload, row, path, root)
                        for row in payload["rows"]
                        if isinstance(row, dict)
                    )
                elif payload.get("experiment") == "hvd_identifiability":
                    rows.append(_normalize_hvd_identifiability(
                        payload, path, root))
                elif (
                    isinstance(payload.get("result"), dict)
                    or (payload.get("method") is not None and payload.get("status") is not None)
                ):
                    rows.append(_normalize_transfer_result(payload, path, root))
                else:
                    errors.append({"path": str(path), "error": "unknown_schema"})
            except Exception as exc:  # Preserve all parse failures in the audit.
                errors.append({"path": str(path), "error": str(exc)[:300]})
    return rows, errors


def _median(values: Iterable[Any]) -> float | None:
    clean = [value for value in (_finite(v) for v in values) if value is not None]
    return statistics.median(clean) if clean else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [value for value in (_finite(v) for v in values) if value is not None]
    return statistics.fmean(clean) if clean else None


def _rate(values: Iterable[Any], *, denominator: int | None = None) -> tuple[int, int, float | None]:
    clean = [value for value in (_boolean(v) for v in values) if value is not None]
    positives = sum(clean)
    total = len(clean) if denominator is None else denominator
    return positives, total, positives / total if total else None


def _failure_aware_median_regret(items: list[dict]) -> tuple[float | None, str]:
    """Median regret with every failed recommendation retained as +infinity."""

    values = []
    for row in items:
        if row.get("true_feasible") is True:
            regret = _finite(row.get("feasible_regret"))
            values.append(math.inf if regret is None else regret)
        else:
            values.append(math.inf)
    if not values:
        return None, "missing"
    median = statistics.median(values)
    if not math.isfinite(median):
        return None, "infinite_due_to_infeasible_recommendations"
    return float(median), "finite"


def extract_trace_rows(rows: list[dict]) -> list[dict]:
    """Flatten compact online traces without exposing checkpoints or models."""

    traces = []
    shared = (
        "run_id", "track", "variant", "method", "implementation",
        "initial_design", "domain", "seed", "d", "N", "n0",
    )
    for row in rows:
        raw_trace = row.get("_online_action_trace")
        if not isinstance(raw_trace, list):
            continue
        base = {field: row.get(field) for field in shared}
        initial_regret = _finite(row.get("initial_best_feasible_regret"))
        traces.append({
            **base,
            "target_call": row.get("n0"),
            "iteration": -1,
            "action_kind": "initial_design_summary",
            "candidate_source": "frozen_initial_design",
            "x_fingerprint": None,
            "selected_score": None,
            "posterior_bayes_risk": None,
            "posterior_theory_margin": None,
            "posterior_constraint_epistemic": None,
            "observed_objective": None,
            "observed_constraint": None,
            "true_objective_post_run": None,
            "true_chance_margin_post_run": None,
            "true_feasible_post_run": row.get("initial_has_true_feasible"),
            "feasible_regret_post_run": initial_regret,
            "incumbent_feasible_regret_post_run": initial_regret,
            "exact_kg_best_new_raw": None,
            "exact_kg_best_replication_raw": None,
            "exact_kg_new_minus_replication_raw": None,
            "truth_join_timing": "post_run_initial_design_audit",
            "target_oracle_used_for_decision": False,
            "result_path": row.get("result_path"),
        })
        for index, action in enumerate(raw_trace):
            if not isinstance(action, dict):
                continue
            observed = action.get("observed_response")
            observed = observed if isinstance(observed, list) else []
            traces.append({
                **base,
                "target_call": _integer(_first(
                    action.get("target_call"),
                    None if row.get("n0") is None else row["n0"] + index + 1,
                )),
                "iteration": _integer(action.get("iteration")),
                "action_kind": action.get("action_kind"),
                "candidate_source": action.get("candidate_source"),
                "x_fingerprint": action.get("x_fingerprint"),
                "selected_score": _finite(action.get("selected_score")),
                "posterior_bayes_risk": _finite(
                    action.get("decision_bayes_risk")),
                "posterior_theory_margin": _finite(
                    action.get("decision_theory_margin")),
                "posterior_constraint_epistemic": _finite(
                    action.get("decision_constraint_epistemic")),
                "observed_objective": _finite(
                    observed[0] if len(observed) > 0 else None),
                "observed_constraint": _finite(
                    observed[1] if len(observed) > 1 else None),
                "true_objective_post_run": _finite(
                    action.get("true_objective_post_run")),
                "true_chance_margin_post_run": _finite(
                    action.get("true_chance_margin_post_run")),
                "true_feasible_post_run": _boolean(
                    action.get("true_feasible_post_run")),
                "feasible_regret_post_run": _finite(
                    action.get("feasible_regret_post_run")),
                "incumbent_feasible_regret_post_run": _finite(
                    action.get("incumbent_feasible_regret_post_run")),
                "exact_kg_best_new_raw": _finite(
                    action.get("exact_kg_best_new_raw")),
                "exact_kg_best_replication_raw": _finite(
                    action.get("exact_kg_best_replication_raw")),
                "exact_kg_new_minus_replication_raw": _finite(
                    action.get("exact_kg_new_minus_replication_raw")),
                "truth_join_timing": action.get("truth_join_timing"),
                "target_oracle_used_for_decision": _boolean(
                    action.get("target_oracle_used_for_decision")),
                "result_path": row.get("result_path"),
            })
    return traces


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    group_fields = (
        "run_id", "track", "variant", "method", "implementation",
        "initial_design", "proposal_mode", "proposal_structural_prior_profile",
        "proposal_source_dimension", "proposal_target_dimension", "domain",
        "d", "N", "n0", "source_calls", "shared_shock_scale",
        "certification_tau", "replicates_per_policy",
    )
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)

    # Add one pooled three-domain row for each otherwise identical cell.
    pooled: dict[tuple, list[dict]] = defaultdict(list)
    pooled_fields = tuple(field for field in group_fields if field != "domain")
    for row in rows:
        pooled[tuple(row.get(field) for field in pooled_fields)].append(row)
    for key, items in pooled.items():
        mapped = dict(zip(pooled_fields, key))
        pooled_key = tuple(mapped.get(field, "ALL" if field == "domain" else None) for field in group_fields)
        groups[pooled_key] = items

    summaries = []
    for key, items in groups.items():
        base = dict(zip(group_fields, key))
        failure_aware_median, failure_aware_status = (
            _failure_aware_median_regret(items))
        feasible_count, feasible_den, feasible_rate = _rate(
            row.get("true_feasible") for row in items)
        initial_count, initial_den, initial_rate = _rate(
            row.get("initial_has_true_feasible") for row in items)
        rescue_eligible = [row for row in items if row.get("initial_has_true_feasible") is False]
        loss_eligible = [row for row in items if row.get("initial_has_true_feasible") is True]
        rescue_count, rescue_den, rescue_rate = _rate(
            (row.get("adaptive_rescue") for row in rescue_eligible),
            denominator=len(rescue_eligible),
        )
        loss_count, loss_den, loss_rate = _rate(
            (row.get("adaptive_loss") for row in loss_eligible),
            denominator=len(loss_eligible),
        )
        improve_count, improve_den, improve_rate = _rate(
            row.get("adaptive_improves_initial_best") for row in loss_eligible)
        nonvacuous_count, nonvacuous_den, nonvacuous_rate = _rate(
            None if row.get("posterior_certificate_vacuous") is None else (
                not row["posterior_certificate_vacuous"]
            )
            for row in items
        )
        posterior_feasible_count, posterior_feasible_den, posterior_feasible_rate = _rate(
            row.get("posterior_feasible") for row in items)
        summaries.append({
            **base,
            "n": len(items),
            "feasible_count": feasible_count,
            "feasible_denominator": feasible_den,
            "feasible_rate": feasible_rate,
            "median_feasible_regret": _median(
                row.get("feasible_regret") for row in items if row.get("true_feasible") is True),
            "mean_feasible_regret": _mean(
                row.get("feasible_regret") for row in items if row.get("true_feasible") is True),
            "failure_aware_median_feasible_regret": failure_aware_median,
            "failure_aware_median_status": failure_aware_status,
            "initial_feasible_count": initial_count,
            "initial_feasible_denominator": initial_den,
            "initial_feasible_rate": initial_rate,
            "median_initial_feasible_regret": _median(
                row.get("initial_best_feasible_regret") for row in items),
            "rescue_count": rescue_count,
            "rescue_denominator": rescue_den,
            "rescue_rate": rescue_rate,
            "adaptive_loss_count": loss_count,
            "adaptive_loss_denominator": loss_den,
            "adaptive_loss_rate": loss_rate,
            "adaptive_improve_count": improve_count,
            "adaptive_improve_denominator": improve_den,
            "adaptive_improve_rate": improve_rate,
            "median_adaptive_regret_change": _median(
                row.get("adaptive_regret_change") for row in items),
            "posterior_feasible_count": posterior_feasible_count,
            "posterior_feasible_denominator": posterior_feasible_den,
            "posterior_feasible_rate": posterior_feasible_rate,
            "nonvacuous_certificate_count": nonvacuous_count,
            "certificate_audit_denominator": nonvacuous_den,
            "nonvacuous_certificate_rate": nonvacuous_rate,
            "false_certificate_count": sum(
                _integer(row.get("false_certificate_count")) or 0 for row in items),
            "certified_point_count": sum(
                _integer(row.get("posterior_certified_count")) or 0
                for row in items),
            "certified_true_feasible_count": sum(
                _integer(row.get("certified_true_feasible_count")) or 0
                for row in items),
            "evaluated_point_count": sum(
                _integer(row.get("evaluated_point_count")) or 0
                for row in items),
            "certified_point_coverage": _safe_ratio(
                sum(_integer(row.get("posterior_certified_count")) or 0
                    for row in items),
                sum(_integer(row.get("evaluated_point_count")) or 0
                    for row in items),
            ),
            "median_certificate_recall": _median(
                row.get("certificate_recall") for row in items),
            "median_minimum_posterior_margin": _median(
                row.get("minimum_posterior_margin") for row in items),
            "median_minimum_true_margin": _median(
                row.get("minimum_true_margin") for row in items),
            "decision_contract_coherent_rate": _mean(
                row.get("decision_contract_coherent")
                for row in items),
            "target_oracle_decision_count": sum(
                row.get("target_oracle_used_for_decision") is True
                for row in items),
            "adaptive_new_point_count": sum(
                _integer(row.get("adaptive_new_point_count")) or 0
                for row in items),
            "adaptive_replication_count": sum(
                _integer(row.get("adaptive_replication_count")) or 0
                for row in items),
            "median_wall_time_sec": _median(row.get("wall_time_sec") for row in items),
            "median_d_over_target_calls": _median(
                row.get("d_over_target_calls") for row in items),
            "median_d_over_total_calls": _median(
                row.get("d_over_total_calls") for row in items),
            "median_log_variance_rmse": _median(
                row.get("log_variance_rmse") for row in items),
            "median_variance_spearman": _median(
                row.get("variance_spearman") for row in items),
            "median_shared_risk_spearman": _median(
                row.get("shared_risk_spearman") for row in items),
            "median_variance_upper_coverage": _median(
                row.get("variance_upper_coverage") for row in items),
            "median_true_feasible_rate": _median(
                row.get("true_feasible_rate") for row in items),
            "median_posterior_feasible_rate_grid": _median(
                row.get("posterior_feasible_rate_grid") for row in items),
            "median_false_feasible_rate": _median(
                row.get("false_feasible_rate") for row in items),
            "median_false_feasible_fraction_of_certified": _median(
                row.get("false_feasible_fraction_of_certified")
                for row in items),
            "median_missed_feasible_rate": _median(
                row.get("missed_feasible_rate") for row in items),
            "median_missed_feasible_fraction_of_true": _median(
                row.get("missed_feasible_fraction_of_true")
                for row in items),
            "median_predicted_true_ratio": _median(
                row.get("median_predicted_true_ratio") for row in items),
            "median_certified_true_ratio": _median(
                row.get("median_certified_true_ratio") for row in items),
            "median_candidate_gen_time_share": _median(
                row.get("candidate_gen_time_share") for row in items),
            "median_kg_compute_time_share": _median(
                row.get("kg_compute_time_share") for row in items),
            "median_posterior_solve_time_share": _median(
                row.get("posterior_solve_time_share") for row in items),
            "median_update_time_share": _median(
                row.get("update_time_share") for row in items),
            "median_simulate_time_share": _median(
                row.get("simulate_time_share") for row in items),
        })
    return sorted(
        summaries,
        key=lambda row: (
            str(row.get("run_id")), str(row.get("track")),
            str(row.get("variant")), str(row.get("domain")),
        ),
    )


def _write_csv(path: Path, rows: list[dict], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, errors = load_rows(args.roots)
    summaries = summarize_rows(rows)
    traces = extract_trace_rows(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "rows.csv", rows, ROW_FIELDS)
    _write_csv(args.out_dir / "grouped_summary.csv", summaries)
    _write_csv(args.out_dir / "traces.csv", traces, TRACE_FIELDS)
    audit = {
        "schema_version": 2,
        "roots": [str(root.resolve()) for root in args.roots],
        "result_json_count": len(rows) + len(errors),
        "parsed_row_count": len(rows),
        "grouped_summary_count": len(summaries),
        "trace_row_count": len(traces),
        "parse_errors": errors,
        "safety_contract": {
            "accepted_filename": "result.json",
            "checkpoint_paths_excluded": True,
            "pickle_or_weight_files_read": False,
            "copies_runtime_artifacts": False,
        },
        "summaries": summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({
        "parsed_rows": len(rows),
        "parse_errors": len(errors),
        "summary_rows": len(summaries),
        "trace_rows": len(traces),
        "out_dir": str(args.out_dir),
    }))


if __name__ == "__main__":
    main()
