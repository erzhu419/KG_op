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
    "decision_backend",
    "structural_prior_profile",
    "hvd_profile",
    "source_discrepancy_update",
    "recheck_top_k",
    "risk_penalty",
    "utility_weight",
    "adaptive_replication_voi",
    "adaptive_replication_count",
    "adaptive_new_point_count",
    "posterior_dominance_enabled",
    "posterior_dominance_switch_count",
    "wall_time_sec",
    "shared_shock_scale",
    "replicates_per_policy",
    "log_variance_rmse",
    "variance_spearman",
    "shared_risk_spearman",
    "variance_upper_coverage",
    "false_feasible_rate",
    "missed_feasible_rate",
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


def _variant_parts(experiment_variant: str, run_id: str) -> tuple[str, str]:
    parts = [part for part in str(experiment_variant or "").split("/") if part]
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
    certificate = _dict(row.get("certificate_outcome_audit"))
    adaptive = _dict(row.get("adaptive_outcome_audit"))
    dominance = _dict(row.get("posterior_dominance"))
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
    source_calls = _integer(_first(
        audit.get("source_simulator_calls"),
        adaptation.get("source_simulator_calls"),
    ))
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
        "decision_backend": row.get("decision_backend"),
        "structural_prior_profile": row.get("structural_prior_profile"),
        "hvd_profile": row.get("hvd_ablation_profile"),
        "source_discrepancy_update": _boolean(row.get("source_discrepancy_update")),
        "recheck_top_k": _integer(row.get("certification_recheck_top_k")),
        "risk_penalty": _finite(row.get("decision_risk_penalty")),
        "utility_weight": _finite(row.get("decision_source_utility_weight")),
        "adaptive_replication_voi": _boolean(row.get("adaptive_replication_voi_enabled")),
        "adaptive_replication_count": _integer(row.get("adaptive_replication_selected_count")),
        "adaptive_new_point_count": _integer(row.get("adaptive_new_point_selected_count")),
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
        "certificate_precision": None,
        "certificate_recall": None,
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
        "replicates_per_policy": _integer(payload.get(
            "replicates_per_policy")),
        "log_variance_rmse": _finite(payload.get("log_variance_rmse")),
        "variance_spearman": _finite(payload.get("variance_spearman")),
        "shared_risk_spearman": _finite(payload.get("shared_risk_spearman")),
        "variance_upper_coverage": _finite(payload.get(
            "variance_upper_coverage")),
        "false_feasible_rate": _finite(payload.get("false_feasible_rate")),
        "missed_feasible_rate": _finite(payload.get("missed_feasible_rate")),
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


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    group_fields = (
        "run_id", "track", "variant", "method", "implementation",
        "initial_design", "proposal_mode", "proposal_structural_prior_profile",
        "proposal_source_dimension", "proposal_target_dimension", "domain",
        "d", "N", "n0", "source_calls", "shared_shock_scale",
        "replicates_per_policy",
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
            "median_certificate_recall": _median(
                row.get("certificate_recall") for row in items),
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
            "median_false_feasible_rate": _median(
                row.get("false_feasible_rate") for row in items),
            "median_missed_feasible_rate": _median(
                row.get("missed_feasible_rate") for row in items),
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "rows.csv", rows, ROW_FIELDS)
    _write_csv(args.out_dir / "grouped_summary.csv", summaries)
    audit = {
        "schema_version": 1,
        "roots": [str(root.resolve()) for root in args.roots],
        "result_json_count": len(rows) + len(errors),
        "parsed_row_count": len(rows),
        "grouped_summary_count": len(summaries),
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
        "out_dir": str(args.out_dir),
    }))


if __name__ == "__main__":
    main()
