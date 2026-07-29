#!/usr/bin/env python3
"""Extract hash-bound compact records beside remote paper results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

try:
    from paper_result_audit import extract_registry_records
except ModuleNotFoundError:
    from .paper_result_audit import extract_registry_records


def _canonical_sha256(payload):
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_record_shard(registry, *, root, origin):
    records, source_receipts = extract_registry_records(
        registry,
        root=root,
        origin=origin,
    )
    return {
        "schema_version": 1,
        "registry_id": registry.get("registry_id"),
        "registry_sha256": _canonical_sha256(registry),
        "origin": str(origin),
        "extracted_at_unix": time.time(),
        "result_root": str(root),
        "record_count": len(records),
        "records_sha256": _canonical_sha256(records),
        "source_receipts": source_receipts,
        "records": records,
        "raw_checkpoints_or_model_weights_read": False,
        "policy_vectors_exported": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    payload = build_record_shard(
        registry,
        root=args.root,
        origin=args.origin,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "origin": payload["origin"],
        "record_count": payload["record_count"],
        "records_sha256": payload["records_sha256"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
