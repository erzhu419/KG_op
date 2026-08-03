#!/usr/bin/env python3
"""Merge distributed convergence shards against one audited result set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from paper_convergence_extract import CONTRACT_ID, CSV_FIELDS
except ModuleNotFoundError:
    from .paper_convergence_extract import CONTRACT_ID, CSV_FIELDS


MERGED_CONTRACT_ID = "post_run_search_convergence_distributed_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(CSV_FIELDS):
            raise ValueError(f"unexpected convergence CSV schema: {path}")
        return list(reader)


def merge_convergence_shards(
    master_audit,
    shard_pairs,
    *,
    track_id,
    master_audit_path=None,
):
    master_records = [
        row for row in master_audit.get("records", ())
        if str(row.get("track_id")) == str(track_id)
    ]
    expected_by_sha = {
        str(row["result_sha256"]): row for row in master_records
    }
    if len(expected_by_sha) != len(master_records):
        raise ValueError("master audit contains duplicate result receipts")

    rows = []
    validations = []
    receipts = []
    observed_shas = set()
    for manifest_path, csv_path in shard_pairs:
        manifest_path = Path(manifest_path)
        csv_path = Path(csv_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract_id") != CONTRACT_ID:
            raise ValueError(
                f"unexpected convergence shard contract: {manifest_path}")
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete convergence shard: {manifest_path}")
        csv_sha256 = _sha256(csv_path)
        if manifest.get("convergence_csv_sha256") != csv_sha256:
            raise ValueError(f"convergence CSV hash mismatch: {csv_path}")
        shard_rows = _read_csv(csv_path)
        if len(shard_rows) != int(manifest.get("trace_row_count", -1)):
            raise ValueError(f"convergence row-count mismatch: {csv_path}")
        shard_validations = list(manifest.get("terminal_validations", ()))
        shard_shas = {
            str(row["result_sha256"]) for row in shard_validations
        }
        if len(shard_shas) != len(shard_validations):
            raise ValueError(
                f"duplicate result receipts inside shard: {manifest_path}")
        duplicate_shas = observed_shas & shard_shas
        if duplicate_shas:
            raise ValueError(
                "convergence shards overlap result receipts: "
                f"{sorted(duplicate_shas)}"
            )
        row_shas = {str(row["result_sha256"]) for row in shard_rows}
        if row_shas != shard_shas:
            raise ValueError(
                f"convergence rows and validations disagree: {csv_path}")
        observed_shas.update(shard_shas)
        rows.extend(shard_rows)
        validations.extend(shard_validations)
        receipts.append({
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha256,
            "result_count": len(shard_validations),
            "trace_row_count": len(shard_rows),
        })

    expected_shas = set(expected_by_sha)
    missing = sorted(expected_shas - observed_shas)
    unexpected = sorted(observed_shas - expected_shas)
    if missing or unexpected:
        raise ValueError(
            "distributed convergence does not exactly cover the master audit: "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_rows = sum(
        int(row.get("target_search_calls") or 0)
        for row in master_records
    )
    if len(rows) != expected_rows:
        raise ValueError(
            "distributed convergence trace count differs from audited calls: "
            f"{len(rows)} != {expected_rows}"
        )
    rows.sort(key=lambda row: (
        str(row["track_id"]),
        str(row["method_identity"]),
        str(row["domain"]),
        int(row["target_dimension"]),
        int(row["seed"]),
        int(row["target_call"]),
    ))
    validations.sort(key=lambda row: (
        str(row["method_identity"]),
        str(row["domain"]),
        int(row["seed"]),
    ))
    manifest = {
        "schema_version": 1,
        "contract_id": MERGED_CONTRACT_ID,
        "source_contract_id": CONTRACT_ID,
        "status": "complete",
        "track_id": str(track_id),
        "result_count": len(master_records),
        "trace_row_count": len(rows),
        "expected_trace_row_count": expected_rows,
        "target_truth_used_post_run_only": True,
        "target_truth_used_for_search_or_selection": False,
        "verification_samples_included": False,
        "policy_vectors_exported": False,
        "source_audit_sha256": (
            _sha256(master_audit_path)
            if master_audit_path is not None else None
        ),
        "shard_receipts": receipts,
        "terminal_validations": validations,
        "terminal_validation_failure_count": sum(
            not row.get("passed", False) for row in validations),
        "terminal_validation_max_abs_error": max(
            (float(row["max_abs_error"]) for row in validations),
            default=None,
        ),
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
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()
    if len(args.manifest) != len(args.csv):
        parser.error("--manifest and --csv must appear the same number of times")
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows, manifest = merge_convergence_shards(
        audit,
        list(zip(args.manifest, args.csv)),
        track_id=args.track_id,
        master_audit_path=audit_path,
    )
    _write_csv(args.out_csv, rows)
    manifest["convergence_csv_sha256"] = _sha256(args.out_csv)
    _write_json(args.out_manifest, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "result_count": manifest["result_count"],
        "trace_row_count": manifest["trace_row_count"],
        "out_csv": args.out_csv,
        "out_manifest": args.out_manifest,
    }, indent=2))


if __name__ == "__main__":
    main()
