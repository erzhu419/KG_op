#!/usr/bin/env python3
"""Create a review-facing analysis artifact without duplicating raw rows.

The full analysis outputs retain one row or receipt per experiment cell.  The
frozen-evidence audit already commits those cells by SHA256, so copying them
again into Git adds size without additional audit value.  This module removes
only registered row-level fields and binds the compact summary to the complete
source analysis by hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


OMITTED_ROW_FIELDS = {
    "compact_rows",
    "result_receipts",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_analysis(payload, *, source_path, source_sha256=None):
    """Remove row-level payloads while preserving every aggregate endpoint."""

    source_path = Path(source_path)
    omitted = {}
    aggregate_payload = {}
    for key, value in payload.items():
        if key in OMITTED_ROW_FIELDS:
            omitted[key] = {
                "container_type": type(value).__name__,
                "entry_count": len(value),
            }
        else:
            aggregate_payload[key] = value

    return {
        "schema_version": 1,
        "contract_id": "or_review_compact_analysis_v1",
        "status": payload.get("status", "missing"),
        "failures": list(payload.get("failures", ())),
        "algorithmic_failure_count": int(
            payload.get("algorithmic_failure_count", 0)
        ),
        "source_analysis": {
            "artifact_name": source_path.name,
            "sha256": (
                str(source_sha256)
                if source_sha256 is not None
                else _sha256(source_path)
            ),
            "contract_id": payload.get("contract_id"),
            "schema_version": payload.get("schema_version"),
            "omitted_row_fields": omitted,
        },
        "aggregate_analysis": aggregate_payload,
        "interpretation_contract": {
            "all_aggregate_endpoints_preserved": True,
            "raw_rows_committed_by_source_analysis_sha256": True,
            "raw_result_cells_committed_by_frozen_matrix_roots": True,
            "omitted_fields_are_row_level_only": True,
        },
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
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.analysis)
    payload = json.loads(source.read_text(encoding="utf-8"))
    compact = compact_analysis(payload, source_path=source)
    _atomic_json(args.out, compact)
    print(json.dumps({
        "status": compact["status"],
        "source_contract_id": compact["source_analysis"]["contract_id"],
        "source_sha256": compact["source_analysis"]["sha256"],
        "omitted_row_fields": compact["source_analysis"][
            "omitted_row_fields"
        ],
        "out": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
