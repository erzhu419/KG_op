#!/usr/bin/env python3
"""Fail-closed descriptive analysis of frozen OPSD temporal audits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from performance.audit_external_energy_temporal_blocks import AUDIT_CONTRACT


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_named_path(value):
    label, separator, path = str(value).partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def _parse_named_count(value):
    label, separator, count = str(value).partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError("expected LABEL=COUNT")
    try:
        parsed = int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("COUNT must be an integer") from error
    return label, parsed


def _load_matrix(label, paths, expected_count=None):
    rows = []
    failures = []
    seen = set()
    paths = sorted(map(Path, paths))
    if expected_count is not None and len(paths) != int(expected_count):
        failures.append({
            "kind": "cell_count_mismatch",
            "matrix": label,
            "expected": int(expected_count),
            "observed": len(paths),
        })
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append({
                "kind": "unreadable_json",
                "matrix": label,
                "path": str(path),
                "error": str(error),
            })
            continue
        if row.get("contract_id") != AUDIT_CONTRACT:
            failures.append({
                "kind": "unexpected_contract",
                "matrix": label,
                "path": str(path),
                "observed": row.get("contract_id"),
            })
            continue
        if row.get("status") not in {"complete", "not_certified"}:
            failures.append({
                "kind": "unexpected_status",
                "matrix": label,
                "path": str(path),
                "observed": row.get("status"),
            })
            continue
        key = (
            str(row.get("target_market")),
            int(row.get("target_seed")),
            str(row.get("arm")),
        )
        if key in seen:
            failures.append({
                "kind": "duplicate_cell",
                "matrix": label,
                "path": str(path),
                "key": list(key),
            })
            continue
        seen.add(key)
        if row.get("postdecision_only") is not True:
            failures.append({
                "kind": "not_postdecision_only",
                "matrix": label,
                "path": str(path),
            })
            continue
        if row.get("used_to_modify_method_or_certificate") is not False:
            failures.append({
                "kind": "audit_used_for_modification",
                "matrix": label,
                "path": str(path),
            })
            continue
        source_path = Path(str(row.get("source_result_path", "")))
        if not source_path.is_file():
            failures.append({
                "kind": "source_result_missing",
                "matrix": label,
                "path": str(path),
                "source_result_path": str(source_path),
            })
            continue
        observed_source_hash = _sha256(source_path)
        if observed_source_hash != row.get("source_result_sha256"):
            failures.append({
                "kind": "source_result_hash_mismatch",
                "matrix": label,
                "path": str(path),
                "expected": row.get("source_result_sha256"),
                "observed": observed_source_hash,
            })
            continue
        certified = bool(row.get("independently_certified"))
        if (row["status"] == "complete") != certified:
            failures.append({
                "kind": "certificate_status_mismatch",
                "matrix": label,
                "path": str(path),
            })
            continue
        audit = row.get("temporal_audit")
        if certified:
            if not isinstance(audit, dict):
                failures.append({
                    "kind": "certified_cell_missing_temporal_audit",
                    "matrix": label,
                    "path": str(path),
                })
                continue
            if audit.get("inferential_certificate_claimed") is not False:
                failures.append({
                    "kind": "temporal_audit_claims_certificate",
                    "matrix": label,
                    "path": str(path),
                })
                continue
        compact = {
            "matrix": label,
            "target_market": key[0],
            "target_seed": key[1],
            "arm": key[2],
            "target_region": row.get("target_region"),
            "independently_certified": certified,
            "source_result_sha256": observed_source_hash,
            "audit_result_sha256": _sha256(path),
        }
        if certified:
            compact.update({
                "minimum_chronological_block_feasibility_probability": float(
                    audit[
                        "minimum_chronological_block_feasibility_probability"
                    ]
                ),
                "nonoverlapping_feasibility_probability": float(
                    audit["nonoverlapping_summary"][
                        "feasibility_probability"
                    ]
                ),
                "sampled_feasibility_probability": float(
                    audit["sampled_distribution_summary"][
                        "feasibility_probability"
                    ]
                ),
                "nonoverlapping_window_count": int(
                    audit["nonoverlapping_summary"]["window_count"]
                ),
            })
        rows.append(compact)
    return rows, failures


def _summary(label, rows, required_probability):
    certified = [row for row in rows if row["independently_certified"]]
    block_stable = [
        row for row in certified
        if row["minimum_chronological_block_feasibility_probability"]
        >= required_probability
    ]
    nonoverlap_stable = [
        row for row in certified
        if row["nonoverlapping_feasibility_probability"]
        >= required_probability
    ]
    jointly_stable = [
        row for row in certified
        if row in block_stable and row in nonoverlap_stable
    ]
    return {
        "matrix": label,
        "cell_count": len(rows),
        "originally_certified_count": len(certified),
        "originally_certified_rate": (
            0.0 if not rows else len(certified) / len(rows)
        ),
        "chronological_block_stable_count": len(block_stable),
        "chronological_block_stable_rate_among_certified": (
            None if not certified else len(block_stable) / len(certified)
        ),
        "nonoverlap_stable_count": len(nonoverlap_stable),
        "nonoverlap_stable_rate_among_certified": (
            None if not certified else len(nonoverlap_stable) / len(certified)
        ),
        "joint_descriptive_stability_count": len(jointly_stable),
        "joint_descriptive_stability_rate_among_certified": (
            None if not certified else len(jointly_stable) / len(certified)
        ),
        "median_minimum_chronological_block_probability": (
            None if not certified else float(np.median([
                row[
                    "minimum_chronological_block_feasibility_probability"
                ]
                for row in certified
            ]))
        ),
        "median_nonoverlapping_probability": (
            None if not certified else float(np.median([
                row["nonoverlapping_feasibility_probability"]
                for row in certified
            ]))
        ),
        "required_probability": float(required_probability),
    }


def analyze(matrices, *, expected_counts=None, required_probability=0.95):
    expected_counts = dict(expected_counts or {})
    rows = []
    failures = []
    for label, paths in matrices.items():
        loaded, matrix_failures = _load_matrix(
            label, paths, expected_counts.get(label))
        rows.extend(loaded)
        failures.extend(matrix_failures)
    summaries = [
        _summary(label, [row for row in rows if row["matrix"] == label],
                 required_probability)
        for label in matrices
    ]
    arm_summaries = []
    for label in matrices:
        matrix_rows = [row for row in rows if row["matrix"] == label]
        for arm in sorted({row["arm"] for row in matrix_rows}):
            arm_summaries.append(_summary(
                f"{label}:{arm}",
                [row for row in matrix_rows if row["arm"] == arm],
                required_probability,
            ))
            arm_summaries[-1]["matrix"] = label
            arm_summaries[-1]["arm"] = arm
    return {
        "schema_version": 1,
        "contract_id": "opsd_postdecision_temporal_audit_analysis_v1",
        "status": "complete" if not failures else "incomplete",
        "postdecision_only": True,
        "inferential_certificate_claimed": False,
        "interpretation": (
            "Chronological-block and physically nonoverlapping summaries are "
            "descriptive stability audits of frozen decisions, not replacement "
            "certificates or iid future-calendar claims."
        ),
        "required_feasibility_probability": float(required_probability),
        "row_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "matrix_summaries": summaries,
        "arm_summaries": arm_summaries,
        "compact_rows": rows,
    }


def _atomic_json(path, payload):
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
    parser.add_argument("--matrix", action="append", required=True)
    parser.add_argument("--expected", action="append", default=[])
    parser.add_argument("--required-probability", type=float, default=0.95)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    named_roots = dict(map(_parse_named_path, args.matrix))
    expected = dict(map(_parse_named_count, args.expected))
    matrices = {
        label: sorted(root.glob("cell*.json"))
        for label, root in named_roots.items()
    }
    payload = analyze(
        matrices,
        expected_counts=expected,
        required_probability=args.required_probability,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "row_count": payload["row_count"],
        "failure_count": payload["failure_count"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
