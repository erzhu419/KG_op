#!/usr/bin/env python3
"""Build a compact immutable registry for the final OR evidence package.

The raw audit intentionally records one SHA256 receipt per result cell.  That
is useful for replay but unnecessarily large for a manuscript artifact.  This
module commits each sorted receipt list to one matrix root and records the
hashes and declared status of every downstream analysis artifact.  A registry
is publication-ready only when the upstream audit is publication-ready and all
registered analyses are complete (possibly with explicitly counted algorithm
failures).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ACCEPTED_ANALYSIS_STATUSES = {
    "complete",
    "complete_with_algorithmic_failures",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_root(receipts):
    """Commit an ordered set of path/hash receipts to one SHA256 root."""

    normalized = sorted(
        (str(row["path"]), str(row["sha256"])) for row in receipts
    )
    digest = hashlib.sha256()
    for path, sha256 in normalized:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_named_path(value):
    label, separator, path = str(value).partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def build_registry(
    audit,
    *,
    audit_path,
    specification_path,
    method_specification_path,
    analyses=(),
    repository_commit,
):
    integrity_failures = []
    if audit.get("publication_ready") is not True:
        integrity_failures.append("upstream frozen-evidence audit is incomplete")
    matrices = []
    for matrix in audit.get("matrices", ()):
        receipts = matrix.get("receipts", ())
        if len(receipts) != int(matrix.get("observed_cell_count", -1)):
            integrity_failures.append(
                f"{matrix.get('name')}: receipt count does not match cells"
            )
        matrices.append({
            "name": str(matrix["name"]),
            "status": str(matrix["status"]),
            "relative_glob": str(matrix["relative_glob"]),
            "expected_cell_count": int(matrix["expected_cell_count"]),
            "observed_cell_count": int(matrix["observed_cell_count"]),
            "successful_cell_count": int(matrix["successful_cell_count"]),
            "algorithmic_failure_cell_count": int(
                matrix["algorithmic_failure_cell_count"]
            ),
            "integrity_failure_count": int(matrix["failure_count"]),
            "receipt_count": len(receipts),
            "sorted_receipt_root_sha256": _receipt_root(receipts),
            "algorithmic_failures": list(matrix.get(
                "algorithmic_failures", ()
            )),
        })

    analysis_rows = []
    for label, path in analyses:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as error:
            integrity_failures.append(
                f"analysis {label} is unreadable: {type(error).__name__}: {error}"
            )
            continue
        status = str(payload.get("status"))
        if status not in ACCEPTED_ANALYSIS_STATUSES:
            integrity_failures.append(
                f"analysis {label} has nonfinal status {status!r}"
            )
        failures = payload.get("failures", ())
        if failures:
            integrity_failures.append(
                f"analysis {label} reports {len(failures)} integrity failures"
            )
        analysis_rows.append({
            "label": str(label),
            "path": str(path),
            "sha256": _sha256(path),
            "contract_id": payload.get("contract_id"),
            "source_analysis_contract_id": payload.get(
                "source_analysis", {}
            ).get("contract_id"),
            "source_analysis_sha256": payload.get(
                "source_analysis", {}
            ).get("sha256"),
            "status": status,
            "declared_failure_count": len(failures),
            "algorithmic_failure_count": int(
                payload.get("algorithmic_failure_count", 0)
            ),
        })

    ready = not integrity_failures
    return {
        "schema_version": 1,
        "contract_id": "or_review_compact_evidence_registry_v1",
        "status": (
            "complete_with_algorithmic_failures"
            if ready and int(audit.get("algorithmic_failure_count", 0)) > 0
            else "complete" if ready else "incomplete"
        ),
        "publication_ready": ready,
        "repository_commit": str(repository_commit),
        "frozen_evidence_audit": {
            "path": str(audit_path),
            "sha256": _sha256(audit_path),
            "contract_id": audit.get("contract_id"),
            "specification_id": audit.get("specification_id"),
            "status": audit.get("status"),
            "matrix_count": int(audit.get("matrix_count", 0)),
            "integrity_failure_count": int(audit.get("failure_count", 0)),
            "algorithmic_failure_count": int(
                audit.get("algorithmic_failure_count", 0)
            ),
        },
        "frozen_evidence_specification": {
            "path": str(specification_path),
            "sha256": _sha256(specification_path),
        },
        "method_specification": {
            "path": str(method_specification_path),
            "sha256": _sha256(method_specification_path),
        },
        "matrices": matrices,
        "analyses": analysis_rows,
        "integrity_failures": integrity_failures,
        "interpretation_contract": {
            "raw_result_cells_are_not_embedded": True,
            "matrix_roots_commit_sorted_relative_path_and_cell_sha256": True,
            "algorithmic_failures_are_outcomes_not_missing_cells": True,
            "postdecision_temporal_audits_are_descriptive_not_certificates": True,
            "energy_v2_and_v3_may_not_be_pooled": True,
            "hvd_is_not_part_of_the_candidate_method_identity": True,
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
    parser.add_argument("--audit", required=True)
    parser.add_argument("--specification", required=True)
    parser.add_argument("--method-specification", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument(
        "--analysis", action="append", default=[], type=_parse_named_path,
        help="registered compact analysis as LABEL=PATH",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    payload = build_registry(
        audit,
        audit_path=args.audit,
        specification_path=args.specification,
        method_specification_path=args.method_specification,
        analyses=args.analysis,
        repository_commit=args.repository_commit,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "publication_ready": payload["publication_ready"],
        "matrix_count": len(payload["matrices"]),
        "analysis_count": len(payload["analyses"]),
        "integrity_failure_count": len(payload["integrity_failures"]),
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["publication_ready"] else 2)


if __name__ == "__main__":
    main()
