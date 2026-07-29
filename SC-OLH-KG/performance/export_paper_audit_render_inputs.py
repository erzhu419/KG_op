#!/usr/bin/env python3
"""Export render inputs only from the passed compact paper audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


CONTRACT_ID = "audited_compact_render_input_v1"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _initial_design(track_id):
    track_id = str(track_id)
    if "sobol" in track_id:
        return "common_sobol"
    if (
        "source" in track_id
        or "archive_fair" in track_id
        or "hvd" in track_id
    ):
        return "frozen_source_informed"
    return "registered_track_specific"


def _render_row(record, *, registry_id):
    dimension = record.get("target_dimension")
    search_calls = record.get("target_search_calls")
    verification_calls = record.get("target_verification_calls")
    source_calls = record.get("source_calls")
    target_total = record.get("target_total_calls")
    source_plus_total = record.get("source_plus_target_total_calls")
    return {
        "audit_registry_id": str(registry_id),
        "run_id": str(record["track_id"]),
        "track": str(record["track_id"]),
        "variant": str(record["method_identity"]),
        "method": str(record["method_identity"]),
        "implementation": record.get("implementation") or "",
        "initial_design": _initial_design(record["track_id"]),
        "domain": str(record["domain"]),
        "seed": record.get("seed"),
        "d": dimension,
        "N": search_calls,
        "n0": 10 if search_calls is not None and search_calls >= 10 else "",
        "source_calls": source_calls,
        "search_calls": search_calls,
        "verification_calls": verification_calls,
        "target_total_calls": target_total,
        "total_calls": source_plus_total,
        "d_over_search_calls": (
            None
            if dimension is None or not search_calls
            else float(dimension) / float(search_calls)
        ),
        "d_over_target_total_calls": (
            None
            if dimension is None or not target_total
            else float(dimension) / float(target_total)
        ),
        "d_over_total_calls": (
            None
            if dimension is None or not source_plus_total
            else float(dimension) / float(source_plus_total)
        ),
        "status": str(record["status"]),
        "true_feasible": record.get("true_feasible"),
        "feasible_regret": record.get("feasible_regret"),
        "constraint_violation": record.get("constraint_violation"),
        "terminal_certified": record.get("terminal_certified"),
        "terminal_false_certificate": record.get("false_certificate"),
        "false_certificate_count": int(bool(
            record.get("false_certificate"))),
        "log_variance_rmse": record.get(
            "aleatoric_log_variance_rmse"),
        "variance_rmse": record.get("aleatoric_variance_rmse"),
        "variance_upper_coverage": record.get(
            "aleatoric_upper_coverage"),
        "variance_shape_correlation": record.get(
            "aleatoric_variance_shape_correlation"),
        "target_oracle_used_for_decision": record.get(
            "target_oracle_used_for_selection"),
        "source_oracle_aided": record.get("source_oracle_aided"),
        "terminal_verification_updates_optimizer": record.get(
            "terminal_verification_updates_optimizer"),
        "result_sha256": str(record["result_sha256"]),
    }


def _group_summary(rows):
    grouped = {}
    for row in rows:
        key = (
            row["track"],
            row["method"],
            row["domain"],
            row["d"],
            row["N"],
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, selected in sorted(grouped.items(), key=str):
        successful = [
            row for row in selected if row["status"] == "ok"]
        feasible = [
            row for row in successful if row["true_feasible"] is True]
        certified = [
            row for row in successful
            if row["terminal_certified"] is True
        ]
        regrets = [
            value
            for row in feasible
            if (value := _finite(row["feasible_regret"])) is not None
        ]
        summaries.append({
            "track": key[0],
            "method": key[1],
            "domain": key[2],
            "d": key[3],
            "N": key[4],
            "registered_count": len(selected),
            "successful_count": len(successful),
            "failure_or_timeout_count": len(selected) - len(successful),
            "true_feasible_count": len(feasible),
            "true_feasible_rate": (
                None if not selected else len(feasible) / len(selected)
            ),
            "terminal_certified_count": len(certified),
            "terminal_certified_rate": (
                None if not selected else len(certified) / len(selected)
            ),
            "false_certificate_count": sum(
                bool(row["terminal_false_certificate"])
                for row in selected
            ),
            "median_feasible_regret": (
                None if not regrets else statistics.median(regrets)
            ),
        })
    return summaries


def export(audit, registry, *, audit_path, registry_path,
           rows_path, summary_path, manifest_path):
    if audit.get("status") != "pass":
        raise ValueError("paper audit must pass before render export")
    if audit.get("registry_id") != registry.get("registry_id"):
        raise ValueError("audit and registry identifiers differ")
    render_tracks = list(map(
        str, registry.get("paper_render_tracks", ())))
    if not render_tracks:
        raise ValueError("registry has no paper_render_tracks")
    track_audits = {
        str(row["track_id"]): row
        for row in audit.get("track_audits", ())
    }
    missing = sorted(set(render_tracks) - set(track_audits))
    failed = sorted(
        track for track in render_tracks
        if track_audits.get(track, {}).get("status") != "pass"
    )
    if missing or failed:
        raise ValueError(
            f"render tracks are unavailable: missing={missing}, "
            f"failed={failed}")
    selected = [
        record for record in audit.get("records", ())
        if str(record["track_id"]) in set(render_tracks)
    ]
    rows = [
        _render_row(record, registry_id=registry["registry_id"])
        for record in selected
    ]
    summaries = _group_summary(rows)
    _write_csv(rows_path, rows)
    _write_csv(summary_path, summaries)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "contract_id": CONTRACT_ID,
        "registry_id": str(registry["registry_id"]),
        "audit_sha256": _sha256(audit_path),
        "registry_sha256": _sha256(registry_path),
        "render_tracks": render_tracks,
        "row_count": len(rows),
        "summary_count": len(summaries),
        "failure_or_timeout_count": sum(
            row["status"] != "ok" for row in rows),
        "rows": {
            "path": str(rows_path),
            "sha256": _sha256(rows_path),
        },
        "summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
        },
        "contracts": {
            "source_is_passed_compact_paper_audit": True,
            "unregistered_tracks_excluded": True,
            "failures_and_timeouts_retained": True,
            "reads_runtime_checkpoints": False,
            "reads_pickle_or_model_weights": False,
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(
        _read_json(args.audit),
        _read_json(args.registry),
        audit_path=args.audit,
        registry_path=args.registry,
        rows_path=args.rows,
        summary_path=args.summary,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
