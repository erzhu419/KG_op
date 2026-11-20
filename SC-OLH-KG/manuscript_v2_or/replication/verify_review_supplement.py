#!/usr/bin/env python3
"""Verify the distributed files of the GP-FTA review supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def record(root: Path, relative: str) -> dict:
    path = root / relative
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for data in iter(lambda: stream.read(1 << 20), b""):
            digest.update(data)
    return {"relpath": relative, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify(root: Path) -> dict:
    manifest = json.loads((root / "REPLICATION_MANIFEST.json").read_bytes())
    for item in manifest["files"]:
        if record(root, item["relpath"]) != item:
            raise RuntimeError(f"File differs from the review manifest: {item['relpath']}")
    return manifest


if __name__ == "__main__":
    manifest = verify(Path(__file__).resolve().parent)
    print(json.dumps({"files_verified": len(manifest["files"]),
                      "experiment_execution_sha256": manifest["experiment_execution_sha256"]}, indent=2))
