#!/usr/bin/env python3
"""Reconstruct compact post-run search convergence for the final paper track.

Only ``result.json`` artifacts are read. Runtime checkpoints, model states,
pickles, and verifier samples are deliberately outside this contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.designs import integer_design_fingerprint  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


CONTRACT_ID = "post_run_search_convergence_v1"
FINAL_TRACK_ID = "final_frozen_source_frontend_backend_d1000_n13"
FLOAT_TOLERANCE = 1e-9

CSV_FIELDS = (
    "track_id",
    "method_identity",
    "domain",
    "target_dimension",
    "seed",
    "target_call",
    "phase",
    "action_kind",
    "point_fingerprint",
    "true_objective_post_run",
    "true_chance_margin_post_run",
    "true_feasible_post_run",
    "incumbent_true_objective_post_run",
    "incumbent_feasible_regret_post_run",
    "incumbent_status",
    "problem_contract_fingerprint",
    "result_sha256",
    "target_truth_used_post_run_only",
    "target_truth_used_for_search_or_selection",
    "verification_samples_included",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_result(payload):
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    rows = payload.get("rows")
    if isinstance(rows, list) and len(rows) == 1:
        return rows[0]
    return payload


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _point(value, dimension):
    if not isinstance(value, (list, tuple)):
        raise ValueError("search trace is missing a policy vector")
    point = tuple(int(item) for item in value)
    if len(point) != int(dimension):
        raise ValueError(
            f"policy dimension {len(point)} differs from {int(dimension)}")
    return point


def _problem(problem_contract):
    domain = str(problem_contract["domain"])
    kwargs = {
        "d": int(problem_contract["dimension"]),
        "L": int(problem_contract["L"]),
        "sigma": float(problem_contract["sigma"]),
        "alpha": float(problem_contract["alpha"]),
    }
    if domain == "FactorShockStatePolicyRZDT1":
        kwargs["shared_shock_scale"] = float(
            problem_contract["shared_shock_scale"])
    base = make_problem(domain, **kwargs)
    base.tau = float(problem_contract["tau"])
    return ScalarizedProblem(
        base,
        weights=problem_contract["scalarization_weights"],
    )


def _truth(problem, point):
    objective = float(problem.true_objective(point))
    mean = float(problem.true_constraint_mean(point))
    sigma = float(problem.true_sigma(point)[1])
    z_alpha = NormalDist().inv_cdf(1.0 - float(problem.alpha))
    margin = mean + z_alpha * sigma - float(problem.tau)
    return {
        "objective": objective,
        "chance_margin": float(margin),
        "feasible": bool(margin <= 0.0),
    }


def _stored_terminal_truth(result):
    truth = result.get("optimization_recommendation_truth")
    if not isinstance(truth, dict):
        truth = result
    return {
        "point": truth.get("x_recommended", result.get("x_recommended")),
        "objective": _finite(truth.get(
            "true_objective", result.get("true_objective"))),
        "chance_margin": _finite(truth.get(
            "true_chance_margin", result.get("true_chance_margin"))),
        "feasible": truth.get("true_feasible", result.get("true_feasible")),
        "true_best_objective": _finite(truth.get(
            "true_best_objective", result.get("true_best_objective"))),
        "feasible_regret": _finite(truth.get(
            "feasible_regret",
            truth.get(
                "simple_regret",
                result.get(
                    "feasible_regret",
                    result.get("feasible_simple_regret"),
                ),
            ),
        )),
    }


def _true_best_objective(problem, terminal):
    if terminal["true_best_objective"] is not None:
        return float(terminal["true_best_objective"])
    if (
        terminal["feasible"] is True
        and terminal["objective"] is not None
        and terminal["feasible_regret"] is not None
    ):
        return float(
            terminal["objective"] - terminal["feasible_regret"])
    _, objective = problem.true_best_feasible()
    return float(objective)


def _history_events(payload, result, record):
    dimension = int(record["target_dimension"])
    history = result.get("history")
    if isinstance(history, list) and history:
        return [{
            "target_call": index + 1,
            "action_kind": str(item.get(
                "selection_reason", "evaluated")),
            "point": _point(item.get("x"), dimension),
            "truth": None,
            "target_oracle_used_for_decision": False,
        } for index, item in enumerate(history)]

    initial = result.get("initial_observations")
    if isinstance(initial, list) and initial:
        return [{
            "target_call": index + 1,
            "action_kind": "frozen_initial_design",
            "point": _point(item.get("point"), dimension),
            "truth": None,
            "target_oracle_used_for_decision": False,
        } for index, item in enumerate(initial)]

    config = payload.get("config") or {}
    initial_points = config.get("initial_design_points")
    actions = result.get("online_action_trace")
    if isinstance(initial_points, list) and isinstance(actions, list):
        n0 = int(result.get("n0", config.get("n0", len(initial_points))))
        events = [{
            "target_call": index + 1,
            "action_kind": "frozen_initial_design",
            "point": _point(point, dimension),
            "truth": None,
            "target_oracle_used_for_decision": False,
        } for index, point in enumerate(initial_points[:n0])]
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError("SC online action trace contains a non-object")
            direct_truth = {
                "objective": _finite(
                    action.get("true_objective_post_run")),
                "chance_margin": _finite(
                    action.get("true_chance_margin_post_run")),
                "feasible": action.get("true_feasible_post_run"),
            }
            if (
                direct_truth["objective"] is None
                or direct_truth["chance_margin"] is None
                or direct_truth["feasible"] is None
            ):
                raise ValueError(
                    "SC online action lacks post-run truth diagnostics")
            events.append({
                "target_call": int(action.get(
                    "target_call", n0 + index + 1)),
                "action_kind": str(action.get(
                    "action_kind", "evaluated")),
                "point": None,
                "point_fingerprint": str(
                    action.get("x_fingerprint") or ""),
                "truth": direct_truth,
                "target_oracle_used_for_decision": bool(
                    action.get("target_oracle_used_for_decision", False)),
            })
        return events
    information = payload.get("information_contract") or {}
    frozen_points = information.get("frozen_initial_points")
    truth_audit = result.get("initial_truth_audit") or {}
    truth_rows = truth_audit.get("rows")
    if (
        isinstance(frozen_points, list)
        and isinstance(truth_rows, list)
        and len(frozen_points) == len(truth_rows)
    ):
        events = []
        for index, (raw_point, raw_truth) in enumerate(zip(
            frozen_points,
            truth_rows,
        )):
            point = _point(raw_point, dimension)
            audited_point = _point(
                raw_truth.get("x_recommended"), dimension)
            if point != audited_point:
                raise ValueError(
                    "proposal truth audit does not match the frozen design")
            truth = {
                "objective": _finite(raw_truth.get("true_objective")),
                "chance_margin": _finite(
                    raw_truth.get("true_chance_margin")),
                "feasible": raw_truth.get("true_feasible"),
            }
            if (
                truth["objective"] is None
                or truth["chance_margin"] is None
                or truth["feasible"] is None
            ):
                raise ValueError(
                    "proposal truth audit is missing post-run truth")
            events.append({
                "target_call": index + 1,
                "action_kind": "frozen_initial_design",
                "point": point,
                "truth": truth,
                "target_oracle_used_for_decision": False,
            })
        return events
    raise ValueError("unsupported result schema for convergence extraction")


def _terminal_validation(problem, terminal, dimension):
    point = _point(terminal["point"], dimension)
    reconstructed = _truth(problem, point)
    errors = {}
    for field in ("objective", "chance_margin"):
        stored = terminal[field]
        if stored is None:
            raise ValueError(f"terminal result is missing true {field}")
        errors[field] = abs(float(stored) - reconstructed[field])
    feasible_match = (
        terminal["feasible"] is not None
        and bool(terminal["feasible"]) == reconstructed["feasible"]
    )
    return {
        "point_fingerprint": integer_design_fingerprint([point]),
        "max_abs_error": max(errors.values()),
        "objective_abs_error": errors["objective"],
        "chance_margin_abs_error": errors["chance_margin"],
        "feasible_match": feasible_match,
        "passed": (
            feasible_match
            and max(errors.values()) <= FLOAT_TOLERANCE
        ),
    }


def extract_record_convergence(record):
    path = Path(str(record["path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = _payload_result(payload)
    problem = _problem(record["problem_contract"])
    terminal = _stored_terminal_truth(result)
    validation = _terminal_validation(
        problem, terminal, record["target_dimension"])
    true_best = _true_best_objective(problem, terminal)
    events = _history_events(payload, result, record)
    expected_calls = int(record["target_search_calls"])
    target_calls = [int(event["target_call"]) for event in events]
    if target_calls != list(range(1, expected_calls + 1)):
        raise ValueError(
            "search trace does not cover exactly target calls "
            f"1..{expected_calls}: {target_calls}")

    n0 = int(
        result.get("n0")
        or (payload.get("config") or {}).get("n0")
        or min(expected_calls, 10)
    )
    incumbent = math.inf
    rows = []
    for event in events:
        if event["target_oracle_used_for_decision"]:
            raise ValueError("post-run target truth was used by a decision")
        point = event.get("point")
        truth = event.get("truth") or _truth(problem, point)
        if truth["feasible"]:
            incumbent = min(incumbent, float(truth["objective"]))
        if math.isfinite(incumbent):
            regret = max(float(incumbent - true_best), 0.0)
            incumbent_objective = float(incumbent)
            incumbent_status = "true_feasible_incumbent"
        else:
            regret = None
            incumbent_objective = None
            incumbent_status = "no_true_feasible_incumbent"
        fingerprint = str(event.get("point_fingerprint") or "")
        if not fingerprint:
            fingerprint = integer_design_fingerprint([point])
        rows.append({
            "track_id": str(record["track_id"]),
            "method_identity": str(record["method_identity"]),
            "domain": str(record["domain"]),
            "target_dimension": int(record["target_dimension"]),
            "seed": int(record["seed"]),
            "target_call": int(event["target_call"]),
            "phase": (
                "initial_design"
                if int(event["target_call"]) <= n0
                else "adaptive_search"
            ),
            "action_kind": str(event["action_kind"]),
            "point_fingerprint": fingerprint,
            "true_objective_post_run": float(truth["objective"]),
            "true_chance_margin_post_run": float(
                truth["chance_margin"]),
            "true_feasible_post_run": bool(truth["feasible"]),
            "incumbent_true_objective_post_run": incumbent_objective,
            "incumbent_feasible_regret_post_run": regret,
            "incumbent_status": incumbent_status,
            "problem_contract_fingerprint": str(
                record["problem_contract_fingerprint"]),
            "result_sha256": str(record["result_sha256"]),
            "target_truth_used_post_run_only": True,
            "target_truth_used_for_search_or_selection": False,
            "verification_samples_included": False,
        })
    return rows, validation


def build_convergence(audit, *, track_id=FINAL_TRACK_ID):
    records = [
        record for record in audit.get("records", ())
        if str(record.get("track_id")) == str(track_id)
    ]
    rows = []
    validations = []
    errors = []
    for record in records:
        try:
            trace, validation = extract_record_convergence(record)
            rows.extend(trace)
            validations.append({
                "method_identity": record["method_identity"],
                "domain": record["domain"],
                "seed": record["seed"],
                "result_sha256": record["result_sha256"],
                **validation,
            })
        except Exception as exc:
            errors.append({
                "method_identity": record.get("method_identity"),
                "domain": record.get("domain"),
                "seed": record.get("seed"),
                "result_sha256": record.get("result_sha256"),
                "error": str(exc),
            })
    expected_rows = sum(
        int(record.get("target_search_calls") or 0) for record in records)
    methods = sorted({str(record["method_identity"]) for record in records})
    result_receipts = sorted(str(
        record["result_sha256"]) for record in records)
    manifest = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": (
            "complete"
            if (
                records
                and not errors
                and len(rows) == expected_rows
                and all(item["passed"] for item in validations)
            )
            else "incomplete"
        ),
        "track_id": str(track_id),
        "result_count": len(records),
        "completed_trace_count": len(validations),
        "trace_row_count": len(rows),
        "expected_trace_row_count": expected_rows,
        "method_identities": methods,
        "terminal_validation_failure_count": sum(
            not item["passed"] for item in validations),
        "terminal_validation_max_abs_error": max(
            (item["max_abs_error"] for item in validations),
            default=None,
        ),
        "target_truth_used_post_run_only": True,
        "target_truth_used_for_search_or_selection": False,
        "verification_samples_included": False,
        "policy_vectors_exported": False,
        "result_receipts_sha256": hashlib.sha256(json.dumps(
            result_receipts,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "terminal_validations": validations,
        "errors": errors,
    }
    return rows, manifest


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--track-id", default=FINAL_TRACK_ID)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows, manifest = build_convergence(
        audit, track_id=args.track_id)
    _write_csv(args.out_csv, rows)
    manifest["source_audit_sha256"] = _sha256(audit_path)
    manifest["convergence_csv_sha256"] = _sha256(args.out_csv)
    _write_json(args.out_manifest, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "result_count": manifest["result_count"],
        "trace_row_count": manifest["trace_row_count"],
        "errors": len(manifest["errors"]),
        "out_csv": args.out_csv,
        "out_manifest": args.out_manifest,
    }, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
