#!/usr/bin/env python3
"""Analyze the V37 replicate-aware clustered-HC3 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import analyze_mean_alignment_v36_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v36_gate as base


VARIANT_ALIASES = {
    "v35_sobol_new": "v35_sobol_new",
    "v36_joint_new_only": "v36_joint_new_only",
    "v37_cluster_rep4": "v36_joint_rep4",
    "v37_cluster_rep8": "v36_joint_rep8",
}
DISPLAY_NAMES = {value: key for key, value in VARIANT_ALIASES.items()}


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        experiment = str(payload.get("experiment_variant", ""))
        display = next(
            (name for name in VARIANT_ALIASES
             if f"/{name}/" in f"/{experiment}/"),
            None,
        )
        if display is None:
            continue
        for raw in payload.get("rows", []):
            row = dict(raw)
            row["gate_variant"] = VARIANT_ALIASES[display]
            row["display_variant"] = display
            rows.append(row)
    return rows


def _source_prior(row):
    numerics = list(row.get("gpr_numerics") or [])
    return (
        dict(numerics[1].get("source_parametric_prior") or {})
        if len(numerics) > 1 else {}
    )


def _cluster_contract(rows, display_variant):
    selected = [
        row for row in rows
        if row.get("display_variant") == display_variant
    ]
    priors = [_source_prior(row) for row in selected]
    return bool(
        selected
        and all(
            prior.get("source_mean_sandwich_clustered_replicates", False)
            and int(prior.get(
                "source_mean_sandwich_replicated_cluster_count", 0)) > 0
            and int(prior.get(
                "source_mean_sandwich_maximum_cluster_size", 1)) > 1
            and not prior.get("target_oracle_used_for_misspecification", True)
            for prior in priors
        )
    )


def summarize(rows, expected_seeds=5):
    result = base.summarize(rows, expected_seeds)
    cluster_contracts = {
        display: _cluster_contract(rows, display)
        for display in ("v37_cluster_rep4", "v37_cluster_rep8")
    }
    result["clustered_hc3_contract"] = cluster_contracts

    result["variant_summaries"] = {
        DISPLAY_NAMES.get(name, name): value
        for name, value in result["variant_summaries"].items()
    }
    result["diagnostic_eligible"] = [
        DISPLAY_NAMES.get(name, name)
        for name in result["diagnostic_eligible"]
        if cluster_contracts.get(DISPLAY_NAMES.get(name, name), False)
    ]
    result["promotion_eligible"] = [
        DISPLAY_NAMES.get(name, name)
        for name in result["promotion_eligible"]
        if cluster_contracts.get(DISPLAY_NAMES.get(name, name), False)
    ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
