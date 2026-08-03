import csv
import hashlib
import json

import pytest

from performance.paper_convergence_extract import CONTRACT_ID, CSV_FIELDS
from performance.paper_convergence_merge import merge_convergence_shards


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_shard(tmp_path, name, result_sha, seed):
    csv_path = tmp_path / f"{name}.csv"
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "track_id": "final",
        "method_identity": "method",
        "domain": "Domain",
        "target_dimension": 1000,
        "seed": seed,
        "target_call": 1,
        "result_sha256": result_sha,
    })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    manifest = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "complete",
        "trace_row_count": 1,
        "convergence_csv_sha256": _sha256(csv_path),
        "terminal_validations": [{
            "method_identity": "method",
            "domain": "Domain",
            "seed": seed,
            "result_sha256": result_sha,
            "max_abs_error": 0.0,
            "passed": True,
        }],
    }
    manifest_path = tmp_path / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, csv_path


def test_distributed_convergence_requires_exact_master_receipt_coverage(
        tmp_path):
    first = _write_shard(tmp_path, "first", "a" * 64, 80)
    second = _write_shard(tmp_path, "second", "b" * 64, 81)
    audit = {"records": [
        {
            "track_id": "final",
            "result_sha256": "a" * 64,
            "target_search_calls": 1,
        },
        {
            "track_id": "final",
            "result_sha256": "b" * 64,
            "target_search_calls": 1,
        },
    ]}

    rows, manifest = merge_convergence_shards(
        audit, [second, first], track_id="final")

    assert manifest["status"] == "complete"
    assert manifest["result_count"] == 2
    assert manifest["completed_trace_count"] == 2
    assert manifest["method_identities"] == ["method"]
    expected_receipt = hashlib.sha256(json.dumps(
        ["a" * 64, "b" * 64], separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    assert manifest["result_receipts_sha256"] == expected_receipt
    assert manifest["trace_row_count"] == 2
    assert [int(row["seed"]) for row in rows] == [80, 81]
    assert manifest["policy_vectors_exported"] is False

    with pytest.raises(ValueError, match="exactly cover"):
        merge_convergence_shards(
            audit, [first], track_id="final")


def test_distributed_convergence_rejects_overlapping_shards(tmp_path):
    first = _write_shard(tmp_path, "first", "a" * 64, 80)
    duplicate = _write_shard(tmp_path, "duplicate", "a" * 64, 80)
    audit = {"records": [{
        "track_id": "final",
        "result_sha256": "a" * 64,
        "target_search_calls": 1,
    }]}

    with pytest.raises(ValueError, match="overlap"):
        merge_convergence_shards(
            audit, [first, duplicate], track_id="final")
