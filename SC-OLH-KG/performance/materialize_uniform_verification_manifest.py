#!/usr/bin/env python3
"""Materialize compact, pre-verification shortlists for a uniform audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_payload(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = payload.get("result")
    if not isinstance(result, dict):
        rows = payload.get("rows")
        if isinstance(rows, list) and len(rows) == 1:
            result = rows[0]
        else:
            result = payload
    return payload, result


def _parse_selection(value):
    track, separator, method = str(value).partition("::")
    if not separator or not track or not method:
        raise ValueError(
            "selection must have the form TRACK_ID::METHOD_IDENTITY")
    return track, method


def materialize(
    audit,
    *,
    selections,
    candidate_count=2,
    candidate_budget=128,
    familywise_delta=0.05,
):
    selections = {_parse_selection(value) for value in selections}
    records = [
        row for row in audit["records"]
        if (row["track_id"], row["method_identity"]) in selections
        and row["status"] == "ok"
    ]
    rows = []
    for compact in records:
        payload, result = _result_payload(compact["path"])
        if result.get(
            "terminal_shortlist_frozen_before_truth_metrics"
        ) is not True:
            raise ValueError(
                f"shortlist was not frozen before truth: {compact['path']}")
        shortlist = list(result.get("frozen_terminal_shortlist") or ())
        unique = []
        seen = set()
        for item in shortlist:
            point = tuple(int(value) for value in item["point"])
            if point not in seen:
                seen.add(point)
                unique.append({
                    "shortlist_position": len(unique) + 1,
                    "shortlist_role": str(item.get(
                        "shortlist_role", "unspecified")),
                    "point": list(point),
                    "target_oracle_used": False,
                    "verification_samples_used": False,
                })
            if len(unique) >= int(candidate_count):
                break
        if len(unique) != int(candidate_count):
            raise ValueError(
                f"result lacks {candidate_count} frozen unique policies: "
                f"{compact['path']}")
        if any(
            item.get("target_oracle_used") is True
            or item.get("verification_samples_used") is True
            for item in shortlist[: int(candidate_count)]
        ):
            raise ValueError(
                f"shortlist contains forbidden information: "
                f"{compact['path']}")
        rows.append({
            "cell_id": (
                f"{compact['track_id']}::{compact['method_identity']}::"
                f"{compact['domain']}::d{compact['target_dimension']}::"
                f"seed{compact['seed']}"
            ),
            "source_track_id": compact["track_id"],
            "source_method_identity": compact["method_identity"],
            "uniform_method_identity": (
                f"uniform_verified::{compact['method_identity']}"
            ),
            "domain": compact["domain"],
            "target_dimension": compact["target_dimension"],
            "seed": compact["seed"],
            "source_calls": compact["source_calls"],
            "target_search_calls": compact["target_search_calls"],
            "optimization_calls_excluding_verification": compact[
                "optimization_calls_excluding_verification"
            ],
            "source_archive_fingerprint": compact[
                "source_archive_fingerprint"
            ],
            "initial_design_fingerprint": compact[
                "initial_design_fingerprint"
            ],
            "source_result_sha256": _sha256(compact["path"]),
            "source_result_status": str(payload.get("status") or "ok"),
            "shortlist_frozen_before_truth_metrics": True,
            "target_oracle_used_to_build_shortlist": False,
            "verification_samples_used_to_build_shortlist": False,
            "shortlist": unique,
        })
    expected = {
        selection: sum(
            row["source_track_id"] == selection[0]
            and row["source_method_identity"] == selection[1]
            for row in rows
        )
        for selection in selections
    }
    return {
        "schema_version": 1,
        "contract_id": "uniform_two_policy_external_verifier_v1",
        "status": "frozen",
        "candidate_count": int(candidate_count),
        "candidate_budgets": [
            int(candidate_budget) for _ in range(int(candidate_count))
        ],
        "familywise_delta": float(familywise_delta),
        "method": "normal_quantile_tolerance",
        "shortlist_mode": "uniform_first_certified_then_primary",
        "search_samples_reused": False,
        "posterior_updated_from_verification": False,
        "target_oracle_used_for_selection": False,
        "selection_counts": {
            f"{track}::{method}": int(count)
            for (track, method), count in sorted(expected.items())
        },
        "row_count": len(rows),
        "rows": sorted(rows, key=lambda row: row["cell_id"]),
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
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-count", type=int, default=2)
    parser.add_argument("--candidate-budget", type=int, default=128)
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    payload = materialize(
        audit,
        selections=args.selection,
        candidate_count=args.candidate_count,
        candidate_budget=args.candidate_budget,
        familywise_delta=args.familywise_delta,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "row_count": payload["row_count"],
        "selection_counts": payload["selection_counts"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
