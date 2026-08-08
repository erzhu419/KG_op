#!/usr/bin/env python3
"""Fail-closed audit for frozen experiment matrices.

The auditor is intentionally generic: a JSON specification declares the raw
cell glob, exact cell count, accepted contracts, immutable field values, and
the fields forming a unique matrix key.  It never computes performance
statistics.  Its only job is to prevent a partial, mixed-version, or malformed
matrix from entering a paper analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MISSING = object()


def nested_value(payload, dotted_path):
    """Read a dotted path from nested JSON objects."""

    value = payload
    for component in str(dotted_path).split("."):
        if not isinstance(value, dict) or component not in value:
            return MISSING
        value = value[component]
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def audit_matrix(results_root, matrix_spec):
    """Audit one matrix without interpreting its outcomes."""

    root = Path(results_root)
    paths = sorted(root.glob(str(matrix_spec["relative_glob"])))
    expected_count = int(matrix_spec["expected_count"])
    accepted_contracts = set(map(str, matrix_spec.get("contract_ids", ())))
    required_values = dict(matrix_spec.get("required_values", {}))
    unique_fields = tuple(map(str, matrix_spec.get("unique_key_fields", ())))
    failures = []
    receipts = []
    seen = {}
    accepted_count = 0

    if len(paths) != expected_count:
        failures.append({
            "kind": "cell_count_mismatch",
            "expected": expected_count,
            "observed": len(paths),
        })

    for path in paths:
        relative_path = str(path.relative_to(root))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append({
                "kind": "unreadable_json",
                "path": relative_path,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            continue
        contract = payload.get("contract_id")
        if accepted_contracts and str(contract) not in accepted_contracts:
            failures.append({
                "kind": "unexpected_contract",
                "path": relative_path,
                "expected": sorted(accepted_contracts),
                "observed": contract,
            })
            continue
        row_failed = False
        for field, expected in required_values.items():
            observed = nested_value(payload, field)
            if observed is MISSING or observed != expected:
                failures.append({
                    "kind": "required_value_mismatch",
                    "path": relative_path,
                    "field": field,
                    "expected": expected,
                    "observed": None if observed is MISSING else observed,
                    "field_missing": observed is MISSING,
                })
                row_failed = True
        key_values = []
        for field in unique_fields:
            value = nested_value(payload, field)
            if value is MISSING:
                failures.append({
                    "kind": "unique_key_field_missing",
                    "path": relative_path,
                    "field": field,
                })
                row_failed = True
                break
            key_values.append(value)
        if len(key_values) == len(unique_fields):
            key = tuple(_json_key(value) for value in key_values)
            if key in seen:
                failures.append({
                    "kind": "duplicate_matrix_key",
                    "path": relative_path,
                    "first_path": seen[key],
                    "fields": list(unique_fields),
                    "values": key_values,
                })
                row_failed = True
            else:
                seen[key] = relative_path
        receipts.append({
            "path": relative_path,
            "sha256": _sha256(path),
        })
        if not row_failed:
            accepted_count += 1

    return {
        "name": str(matrix_spec["name"]),
        "relative_glob": str(matrix_spec["relative_glob"]),
        "expected_cell_count": expected_count,
        "observed_cell_count": len(paths),
        "accepted_cell_count": accepted_count,
        "unique_matrix_key_count": len(seen),
        "status": "complete" if not failures else "incomplete",
        "failure_count": len(failures),
        "failures": failures,
        "receipts": receipts,
    }


def audit_spec(results_root, specification):
    matrices = [
        audit_matrix(results_root, matrix)
        for matrix in specification.get("matrices", ())
    ]
    failures = sum(matrix["failure_count"] for matrix in matrices)
    complete = bool(matrices) and all(
        matrix["status"] == "complete" for matrix in matrices)
    return {
        "schema_version": 1,
        "contract_id": "frozen_experiment_evidence_audit_v1",
        "specification_id": specification.get("contract_id"),
        "status": "complete" if complete else "incomplete",
        "publication_ready": bool(complete),
        "matrix_count": len(matrices),
        "failure_count": int(failures),
        "matrices": matrices,
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
    parser.add_argument("--spec", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    specification = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    payload = audit_spec(args.results_root, specification)
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "publication_ready": payload["publication_ready"],
        "matrix_count": payload["matrix_count"],
        "failure_count": payload["failure_count"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["publication_ready"] else 2)


if __name__ == "__main__":
    main()
