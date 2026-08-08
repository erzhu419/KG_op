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
import copy
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


def _merge_mapping(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_specification(path):
    """Load a specification and verify an optional immutable parent overlay."""

    path = Path(path)
    specification = json.loads(path.read_text(encoding="utf-8"))
    parent = specification.get("parent_specification")
    if parent is None:
        return specification
    parent_path = (path.parent / parent["path"]).resolve()
    observed_sha256 = _sha256(parent_path)
    expected_sha256 = str(parent["sha256"])
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "parent specification digest mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    merged = load_specification(parent_path)
    matrices = {
        str(matrix["name"]): copy.deepcopy(matrix)
        for matrix in merged.get("matrices", ())
    }
    for name, override in specification.get("matrix_overrides", {}).items():
        if name not in matrices:
            raise ValueError(f"unknown parent matrix override: {name}")
        matrices[name] = _merge_mapping(matrices[name], override)
    for matrix in specification.get("additional_matrices", ()):
        name = str(matrix["name"])
        if name in matrices:
            raise ValueError(f"duplicate additional matrix: {name}")
        matrices[name] = copy.deepcopy(matrix)
    merged.update({
        key: copy.deepcopy(value)
        for key, value in specification.items()
        if key not in {
            "parent_specification", "matrix_overrides", "additional_matrices"
        }
    })
    merged["matrices"] = list(matrices.values())
    merged["resolved_parent_specification"] = {
        "path": str(parent_path),
        "sha256": observed_sha256,
    }
    return merged


def audit_matrix(results_root, matrix_spec):
    """Audit one matrix without interpreting its outcomes."""

    root = Path(results_root)
    paths = sorted(root.glob(str(matrix_spec["relative_glob"])))
    expected_count = int(matrix_spec["expected_count"])
    accepted_contracts = set(map(str, matrix_spec.get("contract_ids", ())))
    failure_contracts = set(map(
        str, matrix_spec.get("algorithmic_failure_contract_ids", ())))
    required_values = dict(matrix_spec.get("required_values", {}))
    failure_required_values = dict(
        matrix_spec.get("algorithmic_failure_required_values", {}))
    unique_fields = tuple(map(str, matrix_spec.get("unique_key_fields", ())))
    failure_unique_fields = tuple(map(
        str, matrix_spec.get("algorithmic_failure_unique_key_fields", ())))
    failures = []
    algorithmic_failures = []
    receipts = []
    seen = {}
    failure_seen = {}
    accepted_count = 0
    successful_count = 0

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
        algorithmic_failure = str(contract) in failure_contracts
        if (
            accepted_contracts
            and str(contract) not in accepted_contracts
            and not algorithmic_failure
        ):
            failures.append({
                "kind": "unexpected_contract",
                "path": relative_path,
                "expected": sorted(accepted_contracts),
                "observed": contract,
            })
            continue
        active_required_values = (
            failure_required_values if algorithmic_failure else required_values)
        active_unique_fields = (
            failure_unique_fields if algorithmic_failure else unique_fields)
        active_seen = failure_seen if algorithmic_failure else seen
        row_failed = False
        for field, expected in active_required_values.items():
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
        for field in active_unique_fields:
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
        if len(key_values) == len(active_unique_fields):
            key = tuple(_json_key(value) for value in key_values)
            if key in active_seen:
                failures.append({
                    "kind": "duplicate_matrix_key",
                    "path": relative_path,
                    "first_path": active_seen[key],
                    "fields": list(active_unique_fields),
                    "values": key_values,
                    "algorithmic_failure_contract": algorithmic_failure,
                })
                row_failed = True
            else:
                active_seen[key] = relative_path
        receipts.append({
            "path": relative_path,
            "sha256": _sha256(path),
        })
        if not row_failed:
            accepted_count += 1
            if algorithmic_failure:
                algorithmic_failures.append({
                    "path": relative_path,
                    "contract_id": str(contract),
                    "error_type": payload.get("error_type"),
                    "error_message": payload.get("error_message"),
                })
            else:
                successful_count += 1

    complete = not failures
    status = "incomplete"
    if complete:
        status = (
            "complete_with_algorithmic_failures"
            if algorithmic_failures else "complete"
        )

    return {
        "name": str(matrix_spec["name"]),
        "relative_glob": str(matrix_spec["relative_glob"]),
        "expected_cell_count": expected_count,
        "observed_cell_count": len(paths),
        "accepted_cell_count": accepted_count,
        "successful_cell_count": successful_count,
        "algorithmic_failure_cell_count": len(algorithmic_failures),
        "unique_matrix_key_count": len(seen) + len(failure_seen),
        "status": status,
        "failure_count": len(failures),
        "failures": failures,
        "algorithmic_failures": algorithmic_failures,
        "receipts": receipts,
    }


def audit_spec(results_root, specification):
    matrices = [
        audit_matrix(results_root, matrix)
        for matrix in specification.get("matrices", ())
    ]
    failures = sum(matrix["failure_count"] for matrix in matrices)
    algorithmic_failures = sum(
        matrix["algorithmic_failure_cell_count"] for matrix in matrices)
    complete = bool(matrices) and all(
        matrix["status"] in {"complete", "complete_with_algorithmic_failures"}
        for matrix in matrices
    )
    status = "incomplete"
    if complete:
        status = (
            "complete_with_algorithmic_failures"
            if algorithmic_failures else "complete"
        )
    return {
        "schema_version": 1,
        "contract_id": "frozen_experiment_evidence_audit_v1",
        "specification_id": specification.get("contract_id"),
        "status": status,
        "publication_ready": bool(complete),
        "matrix_count": len(matrices),
        "failure_count": int(failures),
        "algorithmic_failure_count": int(algorithmic_failures),
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
    specification = load_specification(args.spec)
    payload = audit_spec(args.results_root, specification)
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "publication_ready": payload["publication_ready"],
        "matrix_count": payload["matrix_count"],
        "failure_count": payload["failure_count"],
        "algorithmic_failure_count": payload["algorithmic_failure_count"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["publication_ready"] else 2)


if __name__ == "__main__":
    main()
